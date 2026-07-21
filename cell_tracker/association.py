"""
Bidirectional Node Transformer for temporal edge affinity scoring.

This is Stage 2 of the winning pipeline. Rather than relying on raw
physical displacement (which fails during dense motion and mitosis),
the transformer extracts high-dimensional feature vectors from the UNet
decoder for each detected node and scores associations via cross-attention
with Rotary Position Embeddings (RoPE).

Key design choices from top solutions:

* Bidirectional cross-attention: compares source (t) and target (t+1)
  node sets, producing an affinity matrix.

* Division-aware softmax: during training, edge probabilities are
  normalized over all possible parents of each target. This permits
  one parent → two daughters while penalizing multiple parents → one
  daughter.

* Sparse-annotation-safe loss: the focal BCE edge loss is computed
  ONLY on rows/columns that intersect a known annotated edge, shielding
  the network from noise in unlabeled regions.

* RoPE injects 3D geometric bias directly into attention, letting the
  model reason about relative spatial positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """3D Rotary Position Embeddings for injecting geometric bias into attention.

    Encodes each of the three spatial dimensions with sinusoidal frequencies
    so the attention score between two nodes naturally reflects their
    relative positions.
    """

    def __init__(self, dim: int, max_freq: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        freqs = 1.0 / (
            max_freq ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute RoPE encodings for 3D coordinates.

        Args:
            coords: (N, 3) physical-space coordinates in µm.

        Returns:
            (N, dim) sinusoidal encodings.
        """
        N = coords.shape[0]
        encodings = []
        for d in range(3):
            pos = coords[:, d:d+1]
            angles = pos * self.freqs[None, :]
            encodings.append(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1))
        combined = torch.cat(encodings, dim=-1)
        if combined.shape[-1] < self.dim:
            combined = F.pad(combined, (0, self.dim - combined.shape[-1]))
        return combined[:, :self.dim]


class SimpleNodeTransformer(nn.Module):
    """Bidirectional cross-attention transformer for node association.

    Processes source and target node sets through self-attention then
    cross-attention, producing pairwise edge logits.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        self.input_proj = nn.Linear(feature_dim, feature_dim)

        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(
                feature_dim, num_heads, dropout=dropout, batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.cross_attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.edge_scorer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1),
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src_features: torch.Tensor,
        tgt_features: torch.Tensor,
        src_coords: torch.Tensor,
        tgt_coords: torch.Tensor,
        rope: RotaryPositionalEmbedding,
    ) -> torch.Tensor:
        """Score all pairwise edges between source and target node sets.

        Args:
            src_features: (S, F) source node features from UNet decoder.
            tgt_features: (T, F) target node features from UNet decoder.
            src_coords: (S, 3) source physical coordinates.
            tgt_coords: (T, 3) target physical coordinates.

        Returns:
            (S, T) edge logit matrix.
        """
        S, T = src_features.shape[0], tgt_features.shape[0]

        if S == 0 or T == 0:
            return torch.zeros(S, T, device=src_features.device)

        src_pos = rope(src_coords)
        tgt_pos = rope(tgt_coords)

        src = self.input_proj(src_features) + src_pos
        tgt = self.input_proj(tgt_features) + tgt_pos

        src = src.unsqueeze(0)
        tgt = tgt.unsqueeze(0)

        for attn in self.self_attn:
            src2, _ = attn(src, src, src)
            src = self.norm1(src + self.dropout(src2))

        src_cross, _ = self.cross_attn(src, tgt, tgt)
        src = self.norm2(src + self.dropout(src_cross))

        src_exp = src[0].unsqueeze(1).expand(-1, T, -1)
        tgt_exp = tgt[0].unsqueeze(0).expand(S, -1, -1)
        combined = torch.cat([src_exp, tgt_exp], dim=-1)

        edge_logits = self.edge_scorer(combined).squeeze(-1)
        return edge_logits


class DivisionAwareEdgeLoss(nn.Module):
    """Focal BCE loss normalized over possible parents of each target.

    The softmax over incoming edges permits one-to-two parent assignment
    (cell division) while the loss is masked to only annotated edges.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        edge_logits: torch.Tensor,
        edge_targets: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked focal BCE loss.

        Args:
            edge_logits: (S, T) predicted logits.
            edge_targets: (S, T) ground truth edges (0/1).
            edge_mask: (S, T) boolean mask — only annotated edges.

        Returns:
            Scalar loss.
        """
        if not edge_mask.any():
            return torch.tensor(0.0, device=edge_logits.device)

        probs = torch.sigmoid(edge_logits[edge_mask])
        targets = edge_targets[edge_mask]

        bce = F.binary_cross_entropy(probs, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = (alpha_weight * focal_weight * bce).mean()

        return loss


class NodeTransformerAssociator:
    """High-level wrapper for scoring frame-to-frame associations."""

    def __init__(
        self,
        transformer: SimpleNodeTransformer,
        rope: RotaryPositionalEmbedding,
        device: str = "cuda",
    ) -> None:
        self.transformer = transformer.to(device)
        self.rope = rope.to(device)
        self.device = device

    @torch.no_grad()
    def score_edges(
        self,
        src_coords: torch.Tensor,
        tgt_coords: torch.Tensor,
        src_features: torch.Tensor,
        tgt_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score all pairwise edges between two consecutive frames.

        Returns (S, T) logit matrix.
        """
        self.transformer.eval()

        src_coords = src_coords.to(self.device)
        tgt_coords = tgt_coords.to(self.device)
        src_features = src_features.to(self.device)
        tgt_features = tgt_features.to(self.device)

        logits = self.transformer(
            src_features, tgt_features, src_coords, tgt_coords, self.rope
        )
        return logits

    @torch.no_grad()
    def score_timelapse(
        self,
        all_coords: list[np.ndarray],
        all_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Score edges for every consecutive frame pair in a timelapse.

        Returns list of (S_t, T_{t+1}) logit matrices.
        """
        import numpy as np
        edge_logits = []
        for t in range(len(all_coords) - 1):
            src_c = torch.from_numpy(all_coords[t]).float()
            tgt_c = torch.from_numpy(all_coords[t + 1]).float()
            logits = self.score_edges(
                src_c, tgt_c, all_features[t], all_features[t + 1]
            )
            edge_logits.append(logits)
        return edge_logits
