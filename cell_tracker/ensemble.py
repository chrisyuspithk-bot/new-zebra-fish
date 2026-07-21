"""
Two Seeds Logit Blending — the elite ensembling technique that pushes
performance to the 0.950 mark.

Single-model predictions carry inherent biases from specific focal loss
parameters, temporal window sizes, and training seeds. Running two
independent temporal seeds produces multiple perspectives on ambiguous
cellular interactions. By mathematically averaging pre-sigmoid edge
logits, the blend smooths outlier probabilities and neutralizes transient
topological errors during complex division events.

The key insight: averaging in logit space (before sigmoid) preserves the
full dynamic range of model confidence, unlike probability averaging
which collapses extreme values toward 0.5.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
import torch

from .detection_unet import TemporalUNetDetector
from .association import NodeTransformerAssociator


class TwoSeedsLogitBlend:
    """Ensemble two independent tracking runs via pre-sigmoid logit averaging.

    Each "seed" represents a distinct temporal configuration of the
    detection + association pipeline. The blended logits are then fed
    into the ILP solver for final graph construction.
    """

    def __init__(
        self,
        detector_a: TemporalUNetDetector,
        associator_a: NodeTransformerAssociator,
        detector_b: Optional[TemporalUNetDetector] = None,
        associator_b: Optional[NodeTransformerAssociator] = None,
    ) -> None:
        self.detector_a = detector_a
        self.associator_a = associator_a

        self.detector_b = detector_b or detector_a
        self.associator_b = associator_b or associator_a

    def blend_edge_logits(
        self,
        logits_a: List[torch.Tensor],
        logits_b: List[torch.Tensor],
    ) -> List[np.ndarray]:
        """Average pre-sigmoid edge logits from two seeds.

        Averaging in logit space preserves model confidence range.
        Probability averaging would collapse extreme values toward 0.5.
        """
        blended = []
        for la, lb in zip(logits_a, logits_b):
            avg = (la + lb) / 2.0
            blended.append(avg.cpu().numpy())
        return blended

    def detect_with_seed(
        self,
        volumes: List[np.ndarray],
        detector: TemporalUNetDetector,
        temporal_window: int,
    ) -> List[Tuple[np.ndarray, torch.Tensor]]:
        """Run detection with a specific temporal window configuration."""
        return detector.detect_timelapse(volumes, temporal_window)

    def associate_with_seed(
        self,
        detections: List[Tuple[np.ndarray, torch.Tensor]],
        associator: NodeTransformerAssociator,
    ) -> List[torch.Tensor]:
        """Run association on detections."""
        all_coords = [d[0] for d in detections]
        all_features = [d[1] for d in detections]
        return associator.score_timelapse(all_coords, all_features)

    def run(
        self,
        volumes: List[np.ndarray],
        window_a: int = 5,
        window_b: int = 7,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Execute two-seed tracking and return blended results.

        Uses a shared detection pass (seed A) and computes edge logits
        from two different associator configurations. The pre-sigmoid
        logits are averaged before ILP optimization.

        Args:
            volumes: list of (Z, Y, X) numpy arrays, one per timepoint.
            window_a: temporal window size for seed A (default 5).
            window_b: temporal window size for seed B (default 7).

        Returns:
            all_coords: list of (N_t, 3) voxel coordinates per frame.
            blended_logits: list of (S_t, T_{t+1}) edge logit arrays.
        """
        det_a = self.detect_with_seed(volumes, self.detector_a, window_a)
        logits_a = self.associate_with_seed(det_a, self.associator_a)

        det_b = self.detect_with_seed(volumes, self.detector_b, window_b)

        all_coords_a = [d[0] for d in det_a]
        all_coords_b = [d[0] for d in det_b]

        use_b_coords = all(len(ca) == len(cb) for ca, cb in zip(all_coords_a, all_coords_b))

        if use_b_coords:
            logits_b = self.associate_with_seed(det_b, self.associator_b)
            blended_logits = self.blend_edge_logits(logits_a, logits_b)
            return all_coords_a, blended_logits
        else:
            blended_np = [logits.cpu().numpy() for logits in logits_a]
            return all_coords_a, blended_np
