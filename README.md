# Biohub Cell Tracking — Multi-Model Ensemble + ILP Tracking

Improved cell tracking pipeline for the [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) Kaggle competition.

Learnings merged from two strong public baselines:
- [HarshitSama's 0.950 baseline](https://www.kaggle.com/code/harshitsama/biohub-0-950-baseline-explained-reproducible) — UNet-Transformer + ILP solver + division augmentation
- [xiaoleilian's 0.969 solution](https://www.kaggle.com/code/xiaoleilian/biohub-ct-mix-divaug) — UNet3D ensemble + Kalman motion + cubic spline gap closing

## Key Improvements over the 0.950 Baseline

| Area | 0.950 Baseline | Improved |
|------|---------------|----------|
| **Model architecture** | UNet-Transformer only | UNet-Transformer + UNet3D ensemble |
| **Ensemble** | Single model | 3-model recall-weighted ensemble |
| **Detection threshold** | Fixed 0.97 | Adaptive based on heatmap sparsity |
| **TTA** | 8-fliprot per model | 8-fliprot (UNet-Transformer) + 8-flip (UNet3D) |
| **Primary tracking** | ILP solver | ILP solver (when support pack available) |
| **Secondary linking** | ILP-only | Kalman-inspired EMA velocity prediction |
| **Gap closing** | None | Cubic Hermite spline interpolation |
| **Refinement** | Peak local max | Center-of-mass refinement at original resolution |
| **Smoothing** | None | Confidence-weighted linefit smoothing |
| **Repair tracking** | Single pass | Two-pass (seeds + candidate detections) |
| **Division augment** | ✓ | ✓ (same technique) |

## Datasets Required

On Kaggle, attach these datasets:
1. **xiaoleilian/biohub-unet3d-weights** — UNet3D model weights (required)
2. **pilkwang/biohub-tracking-support-pack-50ep-v1** — UNet-Transformer weights + ILP solver (optional, enables ILP tracking)

The code gracefully degrades if the support pack is not available.

## Usage

```bash
# On Kaggle: attach datasets above, then run
python main.py
```

## Configuration

Key parameters in `main.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `UNET_THRESH` | 0.15 | Seed detection threshold |
| `CAND_THR` | 0.05 | Candidate threshold for repair |
| `NMS_UM` | 4.0 | NMS radius (microns) |
| `MAX_LINK_UM` | 10.0 | Max linking distance |
| `GAP_DT` | 2 | Max gap frames to close |
| `LINEFIT_WEIGHT` | 0.8 | Smoothing weight |
| `TTA` | True | Test-time augmentation |
| `DIVAUG` | True | Division metric augmentation |
| `REPAIR` | True | Two-pass repair tracking |
| `USE_ILP` | Auto | ILP solver (auto-enabled with support pack) |

## Requirements

- PyTorch >= 2.0
- numpy, pandas, scipy, scikit-image
- zarr, blosc2 (data loading)
- Optional: tracksdata, polars, pyscipopt, ilpy (support pack — for ILP solver)

*This code was created by an AI agent (OpenHands) on behalf of the user.*
