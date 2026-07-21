# Cell Tracker — Winning Architecture for 3D+Time Cell Tracking

Implementation of the top-performing architecture from the CZ Biohub Cell Tracking Challenge, synthesizing techniques from the highest-scoring public notebooks and research.

## Architecture

The pipeline implements the **tracking-by-detection** paradigm that dominated the leaderboard:

| Stage | Component | LB Contribution |
|-------|-----------|----------------|
| 0 | Multi-scale DoG (classical baseline) | 0.663 → 0.826 |
| 1 | Temporal 3D U-Net centre detection | 0.826 → 0.886 |
| 2 | Bidirectional Node Transformer + RoPE | 0.886 → 0.909 |
| 3 | ILP global graph optimization | 0.909 → 0.950 |
| 4 | Two Seeds Logit Blending + Gap Repair | final polish |

### Key Design Decisions

- **Anisotropy-aware physical coordinates**: all spatial ops happen in µm, not voxels
- **Weighted BCE loss** with extreme negative down-weighting (~1e-3) for sparse GEFF annotations
- **Division-aware softmax** permitting 1→2 parent assignment in the transformer
- **Distance-banded gap repair** with independent centre-model validation
- **Pre-sigmoid logit blending** preserving model confidence range

## Installation

```bash
pip install -e .

# With ILP support (pyscipopt required for global optimization):
pip install -e ".[ilp]"
```

## Usage

```python
import numpy as np
from cell_tracker import CellTrackingPipeline, PhysicalSpace

pipeline = CellTrackingPipeline(
    physical_scale=np.array([2.0, 0.25, 0.25]),  # dz, dy, dx in µm/voxel
    detector="unet",
    use_ensemble=True,
)

tracks = pipeline.track("/path/to/data.zarr")

# tracks: {track_id: [(frame_idx, (z, y, x)_µm), ...]}
```

### Classical Baseline

```python
pipeline = CellTrackingPipeline(
    physical_scale=np.array([2.0, 0.25, 0.25]),
    detector="dog",  # no GPU, no learned weights
)
tracks = pipeline.track("/path/to/data.zarr")
```

## References

Based on winning strategies documented in the CZ Biohub Cell Tracking Challenge:
- pilkwang's Two Seeds Logit Blend (LB 0.950)
- kaiwalyaatulraut's support pack strategy
- dalloliogm's ILP birth/death cost tuning
- Ultrack (royerlab) and Trackastra architectures
