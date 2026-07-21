"""
Temporal 3D U-Net for candidate cell-centre detection.

This is Stage 1 of the winning deep-learning pipeline. Instead of
processing isolated 2D frames (which are highly susceptible to noise),
the network ingests short temporal windows of 3D volumes and emits a
centre-probability field via a final 1×1×1 convolution.

Key design choices from top solutions:

* Weighted BCE loss: positive and negative voxel masses are normalized
  separately, with extreme down-weighting (~1e-3) on the negative class.
  This prevents the sparse GEFF annotations from punishing the model
  for detecting real but unlabeled cells.

* Conservative detection threshold: typically 0.7-0.9, trading recall
  for precision — the edge Jaccard metric heavily penalizes false
  positive nodes.

* The feature maps from the decoder are preserved for downstream use
  by the Node Transformer associator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class ConvBlock(nn.Module):
    """Double 3D convolution with GroupNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        gn = max(1, min(groups, out_ch))
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(gn, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(gn, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporalUNetEncoder(nn.Module):
    """Encoder path with temporal-context 3D convolutions."""

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool3d(2))
            ch = out_ch

        self.bottleneck = ConvBlock(ch, base_filters * (2 ** depth))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.pools[i](x)
        x = self.bottleneck(x)
        return x, skips


class TemporalUNetDecoder(nn.Module):
    """Decoder path with skip connections, outputs a feature map volume."""

    def __init__(
        self,
        base_filters: int = 32,
        depth: int = 4,
        out_features: int = 64,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth):
            in_ch = base_filters * (2 ** (depth - i))
            skip_ch = base_filters * (2 ** (depth - 1 - i))
            self.upconvs.append(
                nn.ConvTranspose3d(in_ch, skip_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(skip_ch * 2, skip_ch))

        self.head = nn.Conv3d(base_filters, out_features, kernel_size=1)

    def forward(
        self, x: torch.Tensor, skips: List[torch.Tensor]
    ) -> torch.Tensor:
        for i in range(self.depth):
            x = self.upconvs[i](x)
            skip = skips[self.depth - 1 - i]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)

        return self.head(x)


class TemporalUNet(nn.Module):
    """Temporal 3D U-Net: ingests T×C×Z×Y×X and emits a centre logit field.

    The temporal window (T) is processed as channels — each timepoint
    is a separate input channel stacked along dim=1.

    The decoder produces a feature map volume that gets passed through
    a 1×1×1 conv to produce per-voxel centre logits, while the raw
    feature map is preserved for Node Transformer feature extraction.
    """

    def __init__(
        self,
        temporal_window: int = 5,
        base_filters: int = 32,
        depth: int = 4,
        out_features: int = 64,
    ) -> None:
        super().__init__()
        self.temporal_window = temporal_window
        self.encoder = TemporalUNetEncoder(
            in_channels=temporal_window, base_filters=base_filters, depth=depth
        )
        self.decoder = TemporalUNetDecoder(
            base_filters=base_filters, depth=depth, out_features=out_features
        )
        self.center_head = nn.Conv3d(out_features, 1, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, T, Z, Y, X) temporal window of volumes.

        Returns:
            center_logits: (B, 1, Z, Y, X) per-voxel centre probability logits.
            features: (B, F, Z, Y, X) feature map for Node Transformer.
        """
        B, T, Z, Y, X = x.shape
        x = x.view(B, T, Z, Y, X)

        feats, skips = self.encoder(x)
        features = self.decoder(feats, skips)
        center_logits = self.center_head(features)

        return center_logits, features

    def extract_node_features(
        self,
        features: torch.Tensor,
        coords: np.ndarray,
        spatial_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Extract feature vectors for each detected node from the feature map.

        Uses differentiable trilinear interpolation at each node's voxel
        position, matching the competition-proven approach.
        """
        if len(coords) == 0:
            return torch.empty(0, features.shape[1], device=features.device)

        D, H, W = spatial_shape
        norm_coords = torch.from_numpy(coords).float().to(features.device)
        norm_coords[:, 0] = 2.0 * norm_coords[:, 0] / max(D - 1, 1) - 1.0
        norm_coords[:, 1] = 2.0 * norm_coords[:, 1] / max(H - 1, 1) - 1.0
        norm_coords[:, 2] = 2.0 * norm_coords[:, 2] / max(W - 1, 1) - 1.0

        grid = norm_coords[None, :, None, None, :]
        sampled = F.grid_sample(
            features, grid, mode="bilinear", align_corners=True
        )
        return sampled[0, :, :, 0, 0].T


class WeightedBCELoss(nn.Module):
    """Weighted binary cross-entropy with separate positive/negative mass normalization.

    Standard BCE would heavily penalize detecting unannotated cells since
    the GEFF ground truth is sparse. This loss normalizes positive and
    negative voxel contributions independently and applies extreme
    down-weighting to the negative class (default 1e-3).
    """

    def __init__(self, negative_weight: float = 1e-3) -> None:
        super().__init__()
        self.negative_weight = negative_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        pos_loss = bce[pos_mask].mean() if pos_mask.any() else 0.0
        neg_loss = bce[neg_mask].mean() if neg_mask.any() else 0.0

        return pos_loss + self.negative_weight * neg_loss


class TemporalUNetDetector:
    """High-level detector wrapper around TemporalUNet.

    Handles inference-time sliding window, NMS, and thresholding.
    """

    def __init__(
        self,
        model: TemporalUNet,
        physical_space,
        threshold: float = 0.7,
        nms_radius_um: float = 4.0,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self._ps = physical_space
        self.threshold = threshold
        self.nms_radius_um = nms_radius_um
        self.device = device

    @torch.no_grad()
    def detect(
        self,
        temporal_window: torch.Tensor,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Detect cell centres in a temporal window.

        Args:
            temporal_window: (1, T, Z, Y, X) tensor on correct device.

        Returns:
            coords: (N, 3) voxel coordinates.
            node_features: (N, F) feature vectors for association.
        """
        self.model.eval()
        center_logits, features = self.model(temporal_window)

        probs = torch.sigmoid(center_logits[0, 0])
        probs_np = probs.cpu().numpy()

        from .utils import AnisotropyAwareNMS

        nms = AnisotropyAwareNMS(self._ps, self.nms_radius_um)
        coords = nms.extract_peaks(probs_np, self.threshold)

        spatial_shape = tuple(probs_np.shape)
        node_features = self.model.extract_node_features(
            features, coords, spatial_shape
        )

        return coords, node_features

    @torch.no_grad()
    def detect_timelapse(
        self,
        volumes: List[np.ndarray],
        temporal_window_size: int = 5,
    ) -> List[Tuple[np.ndarray, torch.Tensor]]:
        """Run detection across an entire timelapse with sliding temporal windows."""
        T = len(volumes)
        pad = temporal_window_size // 2
        results: List[Tuple[np.ndarray, torch.Tensor]] = []

        for t in range(T):
            window_vols = []
            for dt in range(-pad, pad + 1):
                idx = max(0, min(T - 1, t + dt))
                window_vols.append(volumes[idx])

            window = np.stack(window_vols, axis=0)
            tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
            coords, feats = self.detect(tensor)
            results.append((coords, feats))

        return results
