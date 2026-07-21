"""
Center-Confirmed Marginal Gap Repair for recovering cells lost to "blinking".

Fluorescence microscopy suffers from transient cell disappearance — a cell
momentarily drops below the detection threshold due to camera noise, optical
occlusion, or focal-plane shifts. The ILP handles continuous tracking well,
but single-frame gaps require specialized post-processing.

The approach from pilkwang's top notebooks works in three distance bands:

* High-Confidence Anchor (displacement < d_low): auto-accept the interpolated
  node to preserve continuity.
* Marginal Band (d_low ≤ displacement < d_high): only accept if an independent
  DeepCenterUNet3D model confirms cell presence at the interpolated position.
* Strict Rejection (displacement ≥ d_high): gap is biologically implausible
  for a single time step, reject outright.

The auxiliary centre model acts EXCLUSIVELY as a validator, never generating
its own primary graph structure.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Tuple, Callable

from .utils import linear_interpolate_position


class MarginalGapRepair:
    """Post-ILP gap repair with distance-banded centre-model validation."""

    def __init__(
        self,
        d_low_um: float = 4.0,
        d_high_um: float = 12.0,
        center_threshold: float = 0.5,
        center_model: Optional[Callable] = None,
    ) -> None:
        self.d_low = d_low_um
        self.d_high = d_high_um
        self.center_threshold = center_threshold
        self._center_model = center_model

    def _query_center_prior(
        self, position_um: np.ndarray, frame_idx: int
    ) -> float:
        """Query the independent DeepCenterUNet3D model at a physical position.

        If no centre model is available, returns 0.0 (conservative: reject).
        """
        if self._center_model is None:
            return 0.0
        return float(self._center_model(position_um, frame_idx))

    def _find_gaps(
        self,
        tracks: Dict[int, List[Tuple[int, np.ndarray]]],
    ) -> List[Tuple[int, int, np.ndarray, np.ndarray, int]]:
        """Find single-frame gaps: (track_id, frame_before, pos_before, pos_after, gap_frame)."""
        gaps = []
        for track_id, nodes in tracks.items():
            nodes_sorted = sorted(nodes, key=lambda x: x[0])
            for i in range(len(nodes_sorted) - 1):
                f1, p1 = nodes_sorted[i]
                f2, p2 = nodes_sorted[i + 1]
                if f2 - f1 == 2:
                    gaps.append((track_id, f1, p1, p2, f1 + 1))
        return gaps

    def repair(
        self,
        tracks: Dict[int, List[Tuple[int, np.ndarray]]],
        all_coords: List[np.ndarray],
    ) -> Tuple[
        Dict[int, List[Tuple[int, np.ndarray]]],
        List[np.ndarray],
    ]:
        """Attempt to repair single-frame gaps.

        Args:
            tracks: {track_id: [(frame, physical_position), ...]}
            all_coords: list of (N_t, 3) physical coords per frame.

        Returns:
            Updated tracks and updated all_coords (with interpolated nodes
            appended).
        """
        gaps = self._find_gaps(tracks)
        if not gaps:
            return tracks, all_coords

        new_coords = [list(c) for c in all_coords]

        for track_id, f_before, p_before, p_after, f_gap in gaps:
            p_interp = linear_interpolate_position(p_before, p_after)
            displacement = float(np.linalg.norm(p_after - p_before))

            accepted = False
            if displacement < self.d_low:
                accepted = True
            elif displacement < self.d_high:
                prob = self._query_center_prior(p_interp, f_gap)
                accepted = prob >= self.center_threshold

            if accepted:
                node_idx = len(new_coords[f_gap])
                new_coords[f_gap].append(p_interp)

                nodes = tracks[track_id]
                nodes.append((f_gap, p_interp))

        final_coords = [
            np.array(c, dtype=np.float32) if len(c) > 0
            else np.zeros((0, 3), dtype=np.float32)
            for c in new_coords
        ]

        return tracks, final_coords

    @staticmethod
    def build_tracks_from_edges(
        all_coords: List[np.ndarray],
        edges_by_frame: Dict[int, List[Tuple[int, int]]],
    ) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """Build track dictionaries from frame-by-frame edge assignments.

        Each track is a list of (frame_idx, physical_position).
        Uses union-find on node IDs to merge tracks connected by edges.
        """
        T = len(all_coords)

        label = {}
        for t in range(T):
            for i in range(len(all_coords[t])):
                label[(t, i)] = (t, i)

        def find(n):
            while label[n] != n:
                label[n] = label[label[n]]
                n = label[n]
            return n

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                label[rb] = ra

        for t, edges in edges_by_frame.items():
            for src, tgt in edges:
                if (t, src) in label and (t + 1, tgt) in label:
                    union((t, src), (t + 1, tgt))

        groups: Dict[Tuple[int, int], int] = {}
        next_tid = 0
        tracks: Dict[int, List[Tuple[int, np.ndarray]]] = {}

        for t in range(T):
            for i in range(len(all_coords[t])):
                root = find((t, i))
                if root not in groups:
                    groups[root] = next_tid
                    next_tid += 1
                tid = groups[root]
                tracks.setdefault(tid, []).append((t, all_coords[t][i].copy()))

        return tracks
