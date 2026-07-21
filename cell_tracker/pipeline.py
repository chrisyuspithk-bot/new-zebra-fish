"""
End-to-end cell tracking pipeline orchestrating detection, association,
ILP optimization, gap repair, and optional two-seeds logit blending.

This pipeline implements the winning architecture from the CZ Biohub
Cell Tracking Challenge, synthesizing three computational paradigms:

1. Spatial detection via Temporal 3D U-Net (or classical Multi-scale DoG)
2. Temporal association via bidirectional Node Transformers
3. Global lineage reconstruction via ILP with Marginal Gap Repair

Usage:
    from cell_tracker import CellTrackingPipeline, PhysicalSpace

    pipeline = CellTrackingPipeline(
        physical_scale=np.array([2.0, 0.25, 0.25]),
        detector="unet",
        use_ensemble=True,
    )

    tracks = pipeline.track("/path/to/data.zarr")
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Tuple, Literal

from .utils import PhysicalSpace, load_zarr_timelapse
from .detection_classical import MultiScaleDoGDetector
from .detection_unet import TemporalUNet, TemporalUNetDetector
from .association import (
    SimpleNodeTransformer,
    RotaryPositionalEmbedding,
    NodeTransformerAssociator,
)
from .optimization import ILPOptimizer
from .gap_repair import MarginalGapRepair
from .ensemble import TwoSeedsLogitBlend


class CellTrackingPipeline:
    """Complete tracking-by-detection pipeline with optional ensembling."""

    def __init__(
        self,
        physical_scale: np.ndarray,
        detector: Literal["dog", "unet"] = "unet",
        temporal_window: int = 5,
        detection_threshold: float = 0.7,
        nms_radius_um: float = 4.0,
        max_link_distance_um: float = 20.0,
        birth_cost: float = 10.0,
        death_cost: float = 10.0,
        division_cost: float = 5.0,
        edge_weight: float = 1.0,
        gap_d_low_um: float = 4.0,
        gap_d_high_um: float = 12.0,
        gap_center_threshold: float = 0.5,
        use_ensemble: bool = False,
        ensemble_window_b: int = 7,
        device: str = "cuda",
        unet_base_filters: int = 32,
        unet_depth: int = 4,
        unet_out_features: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ) -> None:
        self._ps = PhysicalSpace(physical_scale)
        self.detector_type = detector
        self.temporal_window = temporal_window
        self.use_ensemble = use_ensemble
        self.ensemble_window_b = ensemble_window_b
        self.device = device

        if detector == "dog":
            self._classical_detector = MultiScaleDoGDetector(self._ps)
        else:
            model = TemporalUNet(
                temporal_window=temporal_window,
                base_filters=unet_base_filters,
                depth=unet_depth,
                out_features=unet_out_features,
            )
            self._unet_detector = TemporalUNetDetector(
                model, self._ps, detection_threshold, nms_radius_um, device
            )

        transformer = SimpleNodeTransformer(
            feature_dim=unet_out_features,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
        )
        rope = RotaryPositionalEmbedding(unet_out_features)
        self._associator = NodeTransformerAssociator(transformer, rope, device)

        self._optimizer = ILPOptimizer(
            birth_cost=birth_cost,
            death_cost=death_cost,
            division_cost=division_cost,
            edge_weight=edge_weight,
            max_distance_um=max_link_distance_um,
        )

        self._gap_repair = MarginalGapRepair(
            d_low_um=gap_d_low_um,
            d_high_um=gap_d_high_um,
            center_threshold=gap_center_threshold,
        )

        if use_ensemble and detector == "unet":
            model_b = TemporalUNet(
                temporal_window=ensemble_window_b,
                base_filters=unet_base_filters,
                depth=unet_depth,
                out_features=unet_out_features,
            )
            detector_b = TemporalUNetDetector(
                model_b, self._ps, detection_threshold, nms_radius_um, device
            )
            self._ensemble = TwoSeedsLogitBlend(
                self._unet_detector, self._associator,
                detector_b, self._associator,
            )
        else:
            self._ensemble = None

    def detect(self, volumes: List[np.ndarray]) -> List[np.ndarray]:
        """Run cell detection on all frames."""
        if self.detector_type == "dog":
            results = self._classical_detector.detect_timelapse(volumes)
            return [r[0] for r in results]
        else:
            results = self._unet_detector.detect_timelapse(
                volumes, self.temporal_window
            )
            return [r[0] for r in results]

    def associate(
        self,
        volumes: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Run detection + association, returning coords and edge logits."""
        if self._ensemble is not None:
            coords, logits = self._ensemble.run(
                volumes, self.temporal_window, self.ensemble_window_b
            )
            return coords, logits

        if self.detector_type == "dog":
            det_results = self._classical_detector.detect_timelapse(volumes)
            coords = [r[0] for r in det_results]

            edge_logits = []
            for t in range(len(coords) - 1):
                src_c = self._ps.voxel_to_physical(coords[t])
                tgt_c = self._ps.voxel_to_physical(coords[t + 1])

                dist = np.linalg.norm(
                    src_c[:, None, :] - tgt_c[None, :, :], axis=-1
                )
                edge_logits.append(-dist)
            return coords, edge_logits

        det_results = self._unet_detector.detect_timelapse(
            volumes, self.temporal_window
        )
        coords = [r[0] for r in det_results]
        features = [r[1] for r in det_results]

        edge_logits = self._associator.score_timelapse(coords, features)
        edge_logits_np = [logits.cpu().numpy() for logits in edge_logits]

        return coords, edge_logits_np

    def track(
        self,
        data_path: str,
        timepoints: Optional[slice] = None,
    ) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """Run the full tracking pipeline on a Zarr timelapse.

        Args:
            data_path: path to Zarr v3 volume (T, Z, Y, X).
            timepoints: optional slice to process a subset of timepoints.

        Returns:
            Dict mapping track_id → list of (frame, physical_µm_position).
        """
        volumes = load_zarr_timelapse(data_path, timepoints)

        coords, edge_logits = self.associate(volumes)

        phys_coords = [self._ps.voxel_to_physical(c) for c in coords]

        edge_logits_np = [
            e if isinstance(e, np.ndarray) else e for e in edge_logits
        ]

        edges = self._optimizer.optimize(phys_coords, edge_logits_np)

        tracks = MarginalGapRepair.build_tracks_from_edges(phys_coords, edges)

        tracks, _ = self._gap_repair.repair(tracks, phys_coords)

        return tracks

    def track_volumes(
        self,
        volumes: List[np.ndarray],
    ) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """Run the full tracking pipeline on in-memory volume list.

        Args:
            volumes: list of (Z, Y, X) numpy arrays.

        Returns:
            Dict mapping track_id → list of (frame, physical_µm_position).
        """
        coords, edge_logits = self.associate(volumes)

        phys_coords = [self._ps.voxel_to_physical(c) for c in coords]

        edge_logits_np = [
            e if isinstance(e, np.ndarray) else e for e in edge_logits
        ]

        edges = self._optimizer.optimize(phys_coords, edge_logits_np)

        tracks = MarginalGapRepair.build_tracks_from_edges(phys_coords, edges)

        tracks, _ = self._gap_repair.repair(tracks, phys_coords)

        return tracks
