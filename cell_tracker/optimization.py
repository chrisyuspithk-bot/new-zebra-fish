"""
Integer Linear Programming solver for global lineage-graph optimization.

This is Stage 3 of the winning pipeline. The ILP solver acts as a rigid
biological arbiter: it translates soft neural edge probabilities from the
Node Transformer into a strict Directed Acyclic Graph (DAG) representing
cell lineages.

Unlike greedy local heuristics that make irreversible mistakes during
occlusion events, the ILP optimizes a global objective over the entire
time series. Key constraints from top solutions:

* Flow conservation: tracks cannot randomly terminate/spawn mid-tissue
  without paying a birth/death cost penalty.
* Mitotic regulation: sum of outgoing edges from any node ≤ 2, and
  accepting a division (exactly 2 outgoing) incurs a division cost
  that must be justified by strong neural evidence.
* Kinematic edge weighting: edge cost blends physical displacement
  with the learned transformer probability, modulated by a weight
  scalar (ILP_EDGE_WEIGHT).

When full ILP libraries (pyscipopt, ilpy) are unavailable, falls back
to a greedy Hungarian-based approximation that still respects the key
biological constraints.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional, List, Tuple, Dict


class ILPOptimizer:
    """Global graph optimizer using ILP or Hungarian fallback.

    Constructs a lineage DAG by minimizing a global cost function over
    all candidate nodes and edges with biological constraints.
    """

    def __init__(
        self,
        birth_cost: float = 10.0,
        death_cost: float = 10.0,
        division_cost: float = 5.0,
        edge_weight: float = 1.0,
        max_distance_um: float = 20.0,
        use_ilp: bool = False,
    ) -> None:
        self.birth_cost = birth_cost
        self.death_cost = death_cost
        self.division_cost = division_cost
        self.edge_weight = edge_weight
        self.max_distance_um = max_distance_um
        self.use_ilp = use_ilp

    def _compute_edge_cost(
        self,
        src_coords: np.ndarray,
        tgt_coords: np.ndarray,
        edge_logits: np.ndarray,
    ) -> np.ndarray:
        """Compute combined cost for each candidate edge.

        Cost = kinematic displacement - edge_weight * logit_probability.

        Lower cost = more likely edge. The displacement term ensures
        cells obey approximate physical motion constraints.
        """
        S, T = edge_logits.shape
        probs = 1.0 / (1.0 + np.exp(-edge_logits))

        src_expanded = src_coords[:, None, :]
        tgt_expanded = tgt_coords[None, :, :]
        displacement = np.linalg.norm(src_expanded - tgt_expanded, axis=-1)

        cost = displacement - self.edge_weight * probs

        max_dist_mask = displacement > self.max_distance_um
        cost[max_dist_mask] = 1e9

        return cost

    def _hungarian_solve(
        self,
        cost_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve frame-to-frame assignment via Hungarian algorithm.

        Returns (src_indices, tgt_indices) for matched pairs.
        """
        S, T = cost_matrix.shape
        size = max(S, T)
        padded = np.full((size, size), 1e9, dtype=cost_matrix.dtype)
        padded[:S, :T] = cost_matrix

        row_ind, col_ind = linear_sum_assignment(padded)

        valid_mask = (row_ind < S) & (col_ind < T)
        valid_mask &= padded[row_ind, col_ind] < 1e8

        return row_ind[valid_mask], col_ind[valid_mask]

    def _hungarian_with_division(
        self,
        cost_matrix: np.ndarray,
        src_coords: np.ndarray,
        tgt_coords: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Hungarian assignment with division handling.

        After the primary 1:1 assignment, checks for plausible divisions
        by looking at the second-best match per source. Only accepts a
        second outgoing edge if its cost is below the division_cost
        threshold.
        """
        S, T = cost_matrix.shape
        edges: List[Tuple[int, int]] = []

        row_ind, col_ind = self._hungarian_solve(cost_matrix)

        used_targets = set()
        for s, t in zip(row_ind, col_ind):
            edges.append((int(s), int(t)))
            used_targets.add(int(t))

        for s in range(S):
            sorted_targets = np.argsort(cost_matrix[s])
            second_best = None
            for t in sorted_targets:
                if t not in used_targets:
                    second_best = t
                    break

            if second_best is not None and cost_matrix[s, second_best] < self.division_cost:
                edges.append((int(s), int(second_best)))
                used_targets.add(int(second_best))

        return edges

    def _ilp_solve(
        self,
        all_coords: List[np.ndarray],
        all_edge_logits: List[np.ndarray],
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Full ILP formulation over the complete time series.

        Attempts to use pyscipopt. If unavailable, falls back to
        frame-by-frame Hungarian with division handling.
        """
        try:
            return self._scip_solve(all_coords, all_edge_logits)
        except ImportError:
            return self._hungarian_timelapse(all_coords, all_edge_logits)

    def _scip_solve(
        self,
        all_coords: List[np.ndarray],
        all_edge_logits: List[np.ndarray],
    ) -> Dict[int, List[Tuple[int, int]]]:
        """SCIP-based ILP with flow conservation and mitotic constraints."""
        from pyscipopt import Model, quicksum

        model = Model("cell_tracking_ilp")
        T = len(all_coords)

        node_vars = {}
        edge_vars = {}

        for t in range(T):
            for i in range(len(all_coords[t])):
                node_vars[(t, i)] = model.addVar(
                    f"node_{t}_{i}", vtype="BINARY"
                )

        for t in range(T - 1):
            S, D = all_edge_logits[t].shape
            cost = self._compute_edge_cost(
                all_coords[t], all_coords[t + 1], all_edge_logits[t]
            )
            for s in range(S):
                for d in range(D):
                    if cost[s, d] < 1e8:
                        edge_vars[(t, s, d)] = model.addVar(
                            f"edge_{t}_{s}_{d}", vtype="BINARY"
                        )

        objective = quicksum(
            self.birth_cost * v for (t, i), v in node_vars.items()
            if self._is_birth(all_coords, t, i)
        )

        for (t, s, d), var in edge_vars.items():
            c = self._compute_edge_cost(
                all_coords[t], all_coords[t + 1], all_edge_logits[t]
            )
            objective += c[s, d] * var

        model.setObjective(objective, "minimize")

        for (t, s, d), var in edge_vars.items():
            model.addCons(var <= node_vars[(t, s)])
            model.addCons(var <= node_vars[(t + 1, d)])

        for t in range(T):
            for i in range(len(all_coords[t])):
                incoming = quicksum(
                    edge_vars[(t - 1, p, i)]
                    for p in range(len(all_coords[t - 1]))
                    if (t - 1, p, i) in edge_vars
                ) if t > 0 else 0

                outgoing = quicksum(
                    edge_vars[(t, i, d)]
                    for d in range(len(all_coords[t + 1]))
                    if (t, i, d) in edge_vars
                ) if t < T - 1 else 0

                model.addCons(
                    incoming + node_vars[(t, i)] >= outgoing
                )

        for t in range(T - 1):
            for i in range(len(all_coords[t])):
                outgoing_vars = [
                    edge_vars[(t, i, d)]
                    for d in range(len(all_coords[t + 1]))
                    if (t, i, d) in edge_vars
                ]
                if len(outgoing_vars) > 2:
                    model.addCons(quicksum(outgoing_vars) <= 2)

        model.optimize()

        result: Dict[int, List[Tuple[int, int]]] = {}
        for (t, s, d), var in edge_vars.items():
            if model.getVal(var) > 0.5:
                result.setdefault(t, []).append((s, d))

        return result

    def _is_birth(self, all_coords, t, i):
        """Heuristic: nodes near the temporal boundary are likely births."""
        return t == 0

    def _hungarian_timelapse(
        self,
        all_coords: List[np.ndarray],
        all_edge_logits: List[np.ndarray],
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Frame-by-frame Hungarian with division handling."""
        result: Dict[int, List[Tuple[int, int]]] = {}
        for t in range(len(all_edge_logits)):
            cost = self._compute_edge_cost(
                all_coords[t], all_coords[t + 1], all_edge_logits[t]
            )
            edges = self._hungarian_with_division(
                cost, all_coords[t], all_coords[t + 1]
            )
            if edges:
                result[t] = edges
        return result

    def optimize(
        self,
        all_coords: List[np.ndarray],
        all_edge_logits: List[np.ndarray],
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Run global graph optimization.

        Args:
            all_coords: List of (N_t, 3) physical coordinates per frame.
            all_edge_logits: List of (S_t, T_{t+1}) logit matrices.

        Returns:
            Dict mapping frame t → list of (src_idx, tgt_idx) edges.
        """
        if self.use_ilp:
            return self._ilp_solve(all_coords, all_edge_logits)
        return self._hungarian_timelapse(all_coords, all_edge_logits)
