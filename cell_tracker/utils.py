"""
Anisotropy-aware coordinate transforms, physical-space NMS, and Zarr I/O.

Key insight from winning strategies: all spatial operations must happen in
physical (micrometer) space, not raw voxel space. The optical acquisition
samples the z-axis at significantly lower resolution than x/y, so treating
voxel-space Euclidean distances as biologically meaningful corrupts accuracy.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, center_of_mass
from typing import Optional, Tuple, List


class PhysicalSpace:
    """Convert between voxel and physical (µm) coordinates using the scale array.

    The competition data has anisotropic voxels: typical scale is
    [dz, dy, dx] where dz >> dy ≈ dx (e.g. [2.0, 0.25, 0.25] µm/voxel).
    """

    def __init__(self, scale: np.ndarray) -> None:
        if scale.ndim != 1 or scale.shape[0] != 3:
            raise ValueError(f"scale must be 1D length-3 array, got {scale.shape}")
        self.scale = scale.astype(np.float32)
        self.inv_scale = 1.0 / self.scale

    def voxel_to_physical(self, coords: np.ndarray) -> np.ndarray:
        """Convert voxel [z, y, x] coordinates to physical µm."""
        return coords.astype(np.float32) * self.scale

    def physical_to_voxel(self, coords: np.ndarray) -> np.ndarray:
        """Convert physical µm [z, y, x] coordinates to voxel."""
        return coords.astype(np.float32) * self.inv_scale

    def physical_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """L2 distance in physical µm between two point sets."""
        a_phys = self.voxel_to_physical(a)
        b_phys = self.voxel_to_physical(b)
        return np.linalg.norm(a_phys - b_phys, axis=-1)

    def matching_radius_voxels(self, radius_um: float) -> float:
        """Convert a physical matching radius (µm) to an approximate voxel radius.

        Uses the mean scale for a reasonable isotropic approximation of
        the anisotropic voxel dimensions.
        """
        mean_scale = float(np.mean(self.scale))
        return radius_um / mean_scale


class AnisotropyAwareNMS:
    """Non-maximum suppression with anisotropy-aware pooling kernels.

    Instead of using isotropic 3D max-pooling (which would distort z-axis
    behavior), the kernel dimensions are computed in physical space and
    then converted back to voxel counts per axis.
    """

    def __init__(
        self,
        physical_space: PhysicalSpace,
        radius_um: float = 4.0,
    ) -> None:
        self._ps = physical_space
        self.radius_um = radius_um
        voxel_radii = physical_space.physical_to_voxel(
            np.array([radius_um, radius_um, radius_um])
        )
        self.kernel_size = tuple(max(3, int(2 * r + 1)) for r in voxel_radii)

    def suppress(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply anisotropy-aware max filter and return suppressed heatmap.

        Returns a boolean mask where 1 = local maximum retained.
        """
        kernel = np.ones(self.kernel_size, dtype=heatmap.dtype)
        local_max = maximum_filter(heatmap, footprint=kernel, mode="constant")
        return (heatmap == local_max) & (heatmap > 0)

    def extract_peaks(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return voxel coordinates [N, 3] of peaks above threshold."""
        suppressed = self.suppress(heatmap)
        coords = np.argwhere(suppressed & (heatmap >= threshold))
        return coords.astype(np.float32)


def refine_centroid(
    volume: np.ndarray,
    coords: np.ndarray,
    window: int = 3,
) -> np.ndarray:
    """Refine detection coordinates via weighted centre of mass.

    For each coordinate, extract a local window around it, subtract the
    local background floor, and compute the intensity-weighted centroid.
    This pushes the initial peak estimate closer to the true biological
    cell centre, critical for satisfying the strict matching radius.
    """
    refined = np.zeros_like(coords)
    half = window // 2

    for i, (z, y, x) in enumerate(coords.astype(int)):
        z0, z1 = max(0, z - half), min(volume.shape[0], z + half + 1)
        y0, y1 = max(0, y - half), min(volume.shape[1], y + half + 1)
        x0, x1 = max(0, x - half), min(volume.shape[2], x + half + 1)

        patch = volume[z0:z1, y0:y1, x0:x1].astype(np.float32)
        floor = np.percentile(patch, 10)
        patch = np.maximum(patch - floor, 0)

        if patch.sum() > 0:
            cm = center_of_mass(patch)
            refined[i] = [
                z0 + cm[0],
                y0 + cm[1],
                x0 + cm[2],
            ]
        else:
            refined[i] = coords[i].astype(np.float32)

    return refined


def load_zarr_timelapse(
    path: str,
    timepoints: Optional[slice] = None,
) -> List[np.ndarray]:
    """Lazy-load a Zarr v3 timelapse volume one timepoint at a time.

    Follows the competition-proven strategy: access one chunk per
    timepoint to keep the memory budget tight while the GPU processes
    inference sequentially.
    """
    import zarr

    store = zarr.open(path, mode="r")
    if timepoints is None:
        timepoints = slice(None)

    frames = []
    for t in range(store.shape[0])[timepoints]:
        frames.append(np.asarray(store[t]))
    return frames


def linear_interpolate_position(
    p_before: np.ndarray,
    p_after: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a midpoint between two 3D positions.

    Used by gap repair to postulate a missing cell position assuming
    constant-velocity motion across a single-frame blink.
    """
    return (p_before + p_after) / 2.0
