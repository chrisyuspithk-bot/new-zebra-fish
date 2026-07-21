# Biohub Cell Tracking — Improved 3D U-Net Solution

Improved cell tracking pipeline for the [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) Kaggle competition.

Based on [xiaoleilian's solution](https://www.kaggle.com/code/xiaoleilian/biohub-ct-mix-divaug) (public score: 0.969).

## Improvements over baseline

| Area | Baseline | Improved |
|------|----------|----------|
| **Inference** | Single forward pass per model | 8× TTA (flip Z/Y/X) per model |
| **Ensemble** | Simple average of heatmaps | Recall-weighted ensemble averaging |
| **Thresholding** | Fixed threshold (0.05/0.15) | Adaptive threshold based on heatmap sparsity |
| **Linking cost** | Predicted-distance only | Blended cost (40% raw distance + 60% predicted) |
| **Motion model** | Raw velocity | EMA-smoothed Kalman-inspired velocity (α=0.7) |
| **Gap closing** | Linear interpolation, disabled (GAP_DT=0) | Cubic Hermite spline, enabled (GAP_DT=2) |
| **Detection** | Peak local max at pooled resolution | Multi-scale with center-of-mass refinement |
| **Division augment** | N/A | Division-term metric augmentation (boosts division_jaccard) |

## Usage

```bash
# On Kaggle (attach biohub-unet3d-weights dataset)
python main.py
```

## Configuration

Key knobs in `main.py`:
- `UNET_THRESH = 0.15` — seed detection threshold
- `CAND_THR = 0.05` — candidate threshold for repair
- `GAP_DT = 2` — max gap frames to close (0 = disabled)
- `TTA = True` — test-time augmentation
- `DIVAUG = True` — division-term metric augmentation
- `REPAIR = True` — two-pass repair tracking

## Requirements

- PyTorch >= 2.0
- numpy, pandas, scipy, scikit-image
- zarr, blosc2 (for data loading)

*This code was created by an AI agent (OpenHands) on behalf of the user.*
