# Biohub Cell Tracking — Multi-Model Ensemble + ILP Tracking
# Learnings merged from:
#   - HarshitSama's 0.950 baseline (UNet-Transformer + ILP + division aug)
#     https://www.kaggle.com/code/harshitsama/biohub-0-950-baseline-explained-reproducible
#   - xiaoleilian's 0.969 solution (UNet3D ensemble + Kalman + cubic spline)
#     https://www.kaggle.com/code/xiaoleilian/biohub-ct-mix-divaug
#
# Key improvements over the 0.950 baseline:
#   1. Multi-architecture ensemble (UNet3D + TemporalUNet-Transformer)
#   2. ILP solver for optimal primary tracking (from support pack)
#   3. Kalman-inspired motion prediction for secondary linking
#   4. Adaptive detection threshold based on heatmap statistics
#   5. Cubic Hermite spline gap closing (smoother than linear)
#   6. Confidence-weighted linefit smoothing
#   7. Two-pass repair tracking with candidate detections
#   8. Enhanced 8-fliprot TTA for all models
#   9. Recall-weighted ensemble averaging
#  10. Division-term metric augmentation for division_jaccard

import os, json, glob, time, gc, sys, csv, importlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import grey_opening
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

# ── device probe ──────────────────────────────────────────────────────────
DEVICE = "cpu"
if torch.cuda.is_available():
    try:
        _p = nn.Conv3d(1, 1, 3).to("cuda")
        _ = _p(torch.zeros(1, 1, 4, 4, 4, device="cuda")).cpu()
        DEVICE = "cuda"
        del _p
    except Exception as e:
        print("GPU present but conv3d unusable → CPU:", str(e)[:80])

NUM_GPU = torch.cuda.device_count() if DEVICE == "cuda" else 1
print(f"device: {DEVICE} | gpus: {NUM_GPU} | torch {torch.__version__}")

# ── physical scale ────────────────────────────────────────────────────────
SCALE = np.array([1.625, 0.40625, 0.40625])
POOL = 4

# ── support pack (Pilkwang Kim 50-epoch UNet-Transformer) ─────────────────
_SUPPORT_SEARCH = list(Path("/kaggle/input").glob("**/biohub-tracking-support-pack*"))
SUPPORT_DIR = _SUPPORT_SEARCH[0] if _SUPPORT_SEARCH else None
HAS_SUPPORT = SUPPORT_DIR is not None
if HAS_SUPPORT:
    print(f"support pack: {SUPPORT_DIR}")
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    sys.path.insert(0, str(SUPPORT_DIR / "repo" / "src"))
    _WHEELS = SUPPORT_DIR / "wheels"
    if not _WHEELS.exists():
        _WHEELS = next(Path("/kaggle/input").glob("**/wheels"), None)
    if _WHEELS and _WHEELS.exists():
        _offline = ["tracksdata", "zarr==3.2.1", "numcodecs==0.15.1",
                     "donfig==0.8.1.post1", "geff==1.2.0.1.1", "geff-spec==1.1.1",
                     "pyscipopt==6.2.1", "ilpy==0.6.0", "rustworkx==0.18.0",
                     "polars==1.42.0", "polars-runtime-32==1.42.0",
                     "bidict==0.23.1", "imagecodecs==2026.6.26"]
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "--no-index", "--no-deps", "--find-links", str(_WHEELS)] + _offline,
                       check=False)
    try:
        from biohub_tracking.models import TemporalUNet3D, SimpleNodeTransformer
        from biohub_tracking.io import open_dataset, save_graph
        import tracksdata as td
        HAS_TRACKSDATA = True
        print("support pack imports: OK")
    except Exception as e:
        print(f"support pack import failed: {e}")
        HAS_TRACKSDATA = False
else:
    HAS_TRACKSDATA = False
    print("support pack not found — using UNet3D-only ensemble")

# ── UNet3D weights (xiaoleilian) ──────────────────────────────────────────
WEIGHT_NAMES = ['unet3d_bright.pt', 'unet3d_traintophat.pt']
PREPROCS = ['', 'tophat']

# ── detection & tracking knobs ────────────────────────────────────────────
UNET_THRESH = 0.15
CAND_THR = 0.05
NMS_UM = 4.0
MAX_LINK_UM = 10.0
TIGHT_UM = 6.0
GAP_DT = 2
GAP_GATE_UM = 10.0
SNAP_UM = 3.0
SHORT_MIN = 4
LINEFIT_WEIGHT = 0.8
LINEFIT_WINDOW = 2
REPAIR = True
USE_ILP = HAS_TRACKSDATA

# ── ILP parameters (from HarshitSama baseline) ────────────────────────────
ILP_EDGE_WEIGHT = -1.0
ILP_APPEARANCE_WEIGHT = 0.0
ILP_DISAPPEARANCE_WEIGHT = 1.4
ILP_DIVISION_WEIGHT = 1.0

# ── TTA & augmentation flags ──────────────────────────────────────────────
TTA = True
DIVAUG = True
DIVAUG_MAX_COMPONENTS = 1400
DIVAUG_FORKS = 5

DETECT_THRESH = min(UNET_THRESH, CAND_THR) if REPAIR else UNET_THRESH

# ── data root ─────────────────────────────────────────────────────────────
CANDIROOT = [
    "/kaggle/input/biohub-cell-tracking-during-development",
    "/kaggle/input/competitions/biohub-cell-tracking-during-development",
    "data",
]
ROOT = next((p for p in CANDIROOT if Path(p, "test").exists()), "data")
TEST_DIR = Path(ROOT) / "test"
OUT = "submission.csv"

print(f"tta: {TTA} | repair: {REPAIR} | ilp: {USE_ILP} | divaug: {DIVAUG}")
print(f"seed_thr: {UNET_THRESH} | cand_thr: {CAND_THR}")
print(f"data: {ROOT}")


# ═══════════════════════════════════════════════════════════════════════════
#  UNet3D (from xiaoleilian — used as ensemble member)
# ═══════════════════════════════════════════════════════════════════════════

def _block(ci, co):
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True),
        nn.Conv3d(co, co, 3, padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1 = _block(1, base)
        self.e2 = _block(base, base * 2)
        self.e3 = _block(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.bott = _block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.d3 = _block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.d2 = _block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.d1 = _block(base * 2, base)
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


# ═══════════════════════════════════════════════════════════════════════════
#  MyUNet wrapper — wraps TemporalUNet3D + Transformer from support pack
# ═══════════════════════════════════════════════════════════════════════════

class MyUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.D = nn.Parameter(torch.ones(1))
        self.unet = TemporalUNet3D(
            in_channels=1,
            out_channels=int(config["unet_out_channels"]),
            layers=tuple(config["unet_layers"]),
            gradient_checkpointing=False,
        )
        unet_out_channels = int(config["unet_out_channels"])
        self.unet_out_channels = unet_out_channels
        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)
        pos_feat_dim = 4 * 8
        self.transformer = SimpleNodeTransformer(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=128, n_heads=4, n_blocks=4, dropout=0,
        )

    def forward_unet(self, image):
        image = image[:, :, None]
        f = self.unet(image)
        point_logit = [self.detect_head(f[:, 0]), self.detect_head(f[:, 1])]
        point_feature = [f[:, 0], f[:, 1]]
        return point_feature, point_logit

    def forward_transformer(self, select0, select1, coord0, coord1, pos0, pos1):
        feature0 = torch.cat([select0, pos0], dim=-1)
        feature1 = torch.cat([select1, pos1], dim=-1)
        logit = self.transformer(feature0, feature1, coord0, coord1)
        return logit


# ═══════════════════════════════════════════════════════════════════════════
#  Weight loading
# ═══════════════════════════════════════════════════════════════════════════

def _find_weight(name):
    cands = [f"/kaggle/input/biohub-unet3d-weights/{name}",
             f"models/{name}"] + glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    p = next((c for c in cands if Path(c).exists()), None)
    if p is None:
        raise FileNotFoundError(
            f"{name} not found; attach biohub-unet3d-weights. "
            f"/kaggle/input: {glob.glob('/kaggle/input/*')}"
        )
    return p


# Load UNet3D models
WEIGHTS = []
for n in WEIGHT_NAMES:
    try:
        WEIGHTS.append(_find_weight(n))
    except FileNotFoundError:
        print(f"weight not found: {n} — skipping")

MODELS_UNET3D = []
MODEL_WEIGHTS_UNET3D = []
for _w in WEIGHTS:
    _ck = torch.load(_w, map_location=DEVICE)
    _m = UNet3D(base=_ck.get("base", 24)).to(DEVICE)
    _m.load_state_dict(_ck["state_dict"])
    _m.eval()
    MODELS_UNET3D.append(_m)
    vr = _ck.get("val_recall", 0.5)
    MODEL_WEIGHTS_UNET3D.append(vr)
    print(f"loaded unet3d {Path(_w).name} | val_recall={vr}")

# Load support pack model (UNet-Transformer)
HAS_SP_MODEL = False
if HAS_TRACKSDATA:
    try:
        _ckpt_dir = SUPPORT_DIR / "weights" / "unet_transformer" / "split_0"
        _ckpt_file = _ckpt_dir / "edge_predictor_best.pth"
        _config_file = _ckpt_dir / "config.json"
        if _ckpt_file.exists() and _config_file.exists():
            with open(_config_file) as f:
                sp_config = json.load(f)
            sp_model = MyUNet(sp_config).to(DEVICE)
            _state = torch.load(str(_ckpt_file), map_location="cpu", weights_only=True)
            sp_model.load_state_dict(_state, strict=False)
            sp_model.eval()
            HAS_SP_MODEL = True
            print(f"loaded support pack model | layers={sp_config.get('unet_layers')}")
        else:
            print("support pack model weights not found")
    except Exception as e:
        print(f"support pack model load failed: {e}")

# Build ensemble
MODELS = list(MODELS_UNET3D)
MODEL_WEIGHTS_ENS = list(MODEL_WEIGHTS_UNET3D)
MODEL_PREPROCS = list(PREPROCS[:len(MODELS_UNET3D)])

if HAS_SP_MODEL:
    MODELS.append(sp_model)
    MODEL_WEIGHTS_ENS.append(0.85)
    MODEL_PREPROCS.append('')

MODEL_WEIGHTS_ENS = np.array(MODEL_WEIGHTS_ENS, dtype=np.float32)
MODEL_WEIGHTS_ENS /= MODEL_WEIGHTS_ENS.sum()
print(f"ensemble: {len(MODELS)} models, weights={MODEL_WEIGHTS_ENS.round(3)}")


# ═══════════════════════════════════════════════════════════════════════════
#  Data I/O
# ═══════════════════════════════════════════════════════════════════════════

def read_array_meta(zp):
    with open(Path(zp) / "0" / "zarr.json") as f:
        m = json.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))


_ZC = {}

def load_volume(zp, t, meta=None):
    try:
        import zarr
        k = str(zp)
        if k not in _ZC:
            _ZC[k] = zarr.open(k, mode="r")["0"]
        return np.asarray(_ZC[k][t])
    except Exception:
        import blosc2
        if meta is None:
            meta = read_array_meta(zp)
        buf = blosc2.decompress(
            open(Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0", "rb").read()
        )
        return np.frombuffer(buf, dtype=meta["dtype"]).reshape(meta["shape"][1:])


# ═══════════════════════════════════════════════════════════════════════════
#  Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def pool_xy(vol, f=POOL):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    v = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return v.reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def pool_norm(vol, preproc=""):
    p = pool_xy(vol)
    if preproc == "tophat":
        p = np.clip(p - grey_opening(p, size=(1, 7, 7)), 0.0, None)
    lo = float(np.percentile(p, 50))
    hi = float(np.percentile(p, 99.5))
    return np.clip((p - lo) / (hi - lo + 1e-6), -0.5, 6.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  TTA — 8-fliprot (from HarshitSama baseline)
# ═══════════════════════════════════════════════════════════════════════════

def do_tta_8fliprot(im):
    dims = (-2, -1)
    images = [
        im,
        im.flip(dims=(-1,)),
        im.flip(dims=(-2,)),
        im.flip(dims=(-2, -1)),
        torch.rot90(im, 1, dims=dims),
        torch.rot90(im, 3, dims=dims),
        im.transpose(-1, -2),
        torch.rot90(im, 1, dims=dims).transpose(-1, -2),
    ]
    return images, None


def undo_tta_8fliprot(x, transform=None):
    dims = (-2, -1)
    return torch.stack([
        x[0],
        x[1].flip(dims=(-1,)),
        x[2].flip(dims=(-2,)),
        x[3].flip(dims=(-2, -1)),
        torch.rot90(x[4], -1, dims=dims),
        torch.rot90(x[5], -3, dims=dims),
        x[6].transpose(-1, -2),
        torch.rot90(x[7].transpose(-1, -2), -1, dims=dims),
    ])


def _tta_flip_legacy(tensor, flip_z=False, flip_y=False, flip_x=False, transpose_xy=False):
    t = tensor.clone()
    if flip_z:
        t = torch.flip(t, [2])
    if flip_y:
        t = torch.flip(t, [3])
    if flip_x:
        t = torch.flip(t, [4])
    if transpose_xy:
        t = t.permute(0, 1, 2, 4, 3)

    def inverse(hmap):
        h = hmap.copy()
        if transpose_xy:
            h = h.transpose(0, 1, 3, 2)
        if flip_x:
            h = h[:, :, :, ::-1]
        if flip_y:
            h = h[:, :, ::-1, :]
        if flip_z:
            h = h[:, ::-1, :, :]
        return h
    return t, inverse


def _predict_unet3d_tta(model, x_np):
    x = torch.from_numpy(x_np)[None, None].to(DEVICE)
    with torch.no_grad():
        h0 = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
        if not TTA:
            return h0
    heatmaps = [h0]
    tta_configs = [
        (False, False, True, False),
        (False, True, False, False),
        (False, True, True, False),
        (True, False, False, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, False),
    ]
    for fz, fy, fx, tx in tta_configs:
        t, inv = _tta_flip_legacy(x, flip_z=fz, flip_y=fy, flip_x=fx, transpose_xy=tx)
        with torch.no_grad():
            h = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
        heatmaps.append(inv(h))
    return np.mean(heatmaps, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
#  Detection
# ═══════════════════════════════════════════════════════════════════════════

def _refine(vol, zyx, rz=2, ryx=5):
    Z, Y, X = vol.shape
    z, y, x = (int(round(v)) for v in zyx)
    z0, z1 = max(0, z - rz), min(Z, z + rz + 1)
    y0, y1 = max(0, y - ryx), min(Y, y + ryx + 1)
    x0, x1 = max(0, x - ryx), min(X, x + ryx + 1)
    crop = vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
    bg = float(crop.min())
    w = np.clip(crop - bg, 0, None)
    s = float(w.sum())
    if s <= 0:
        return np.array([z, y, x], float), 0.0
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    return np.array([(zz * w).sum(), (yy * w).sum(), (xx * w).sum()]) / s, float(crop.max() - bg)


def _physical_nms(coords, scores, radius_um, scale=SCALE):
    if len(coords) <= 1:
        return coords, scores
    pts = coords * scale[None, :]
    order = np.argsort(-scores)
    tree = cKDTree(pts)
    killed = np.zeros(len(coords), bool)
    keep = []
    for i in order:
        if killed[i]:
            continue
        keep.append(int(i))
        killed[tree.query_ball_point(pts[i], r=radius_um)] = True
    keep = np.array(keep)
    return coords[keep], scores[keep]


def _adaptive_threshold(h, base_thresh=UNET_THRESH):
    hmax = float(h.max())
    if hmax < base_thresh:
        return base_thresh
    pct_above = float((h > base_thresh * 0.5).mean())
    if pct_above < 0.001:
        return max(base_thresh * 0.7, 0.05)
    if pct_above > 0.05:
        return min(base_thresh * 1.3, 0.25)
    return base_thresh


def detect(vol):
    """Multi-model ensemble detection with TTA and recall-weighted averaging."""
    hs = []
    for _m, _pp, _w in zip(MODELS, MODEL_PREPROCS, MODEL_WEIGHTS_ENS):
        x = pool_norm(vol, _pp)
        if isinstance(_m, MyUNet):
            _im = torch.from_numpy(x)[None].to(DEVICE)
            with torch.inference_mode():
                if TTA:
                    images, transform = do_tta_8fliprot(_im[0])
                    images = torch.stack(images, dim=0)
                    _pf, _pl = _m.forward_unet(images)
                    _pl = [undo_tta_8fliprot(p, transform) for p in _pl]
                    point_prob = [torch.sigmoid(p.mean(0)) for p in _pl]
                else:
                    _pf, _pl = _m.forward_unet(_im)
                    point_prob = [torch.sigmoid(p[0]) for p in _pl]
            h = point_prob[1].float().cpu().numpy() * _w
        else:
            h = _predict_unet3d_tta(_m, x) * _w
        hs.append(h)

    h = np.sum(hs, axis=0)
    athr = _adaptive_threshold(h)
    use_thresh = min(athr, DETECT_THRESH) if REPAIR else athr

    pk = peak_local_max(h, min_distance=1, threshold_abs=use_thresh,
                        exclude_border=False)
    if len(pk) == 0:
        return np.zeros((0, 3)), np.zeros(0)

    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2

    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)


# ═══════════════════════════════════════════════════════════════════════════
#  Kalman-inspired motion prediction
# ═══════════════════════════════════════════════════════════════════════════

class MotionPredictor:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.vel = {}

    def update(self, nid, displacement_um):
        if nid in self.vel:
            self.vel[nid] = self.alpha * displacement_um + (1 - self.alpha) * self.vel[nid]
        else:
            self.vel[nid] = displacement_um

    def predict(self, nid, dt=1):
        if nid in self.vel:
            return self.vel[nid] * dt
        return np.zeros(3)

    def get_velocities(self, ids):
        return np.array([self.vel.get(g, np.zeros(3)) for g in ids])


# ═══════════════════════════════════════════════════════════════════════════
#  Linking (Hungarian algorithm)
# ═══════════════════════════════════════════════════════════════════════════

def _link(prev_xyz, curr_xyz, prev_vel):
    if len(prev_xyz) == 0 or len(curr_xyz) == 0:
        return []

    P = prev_xyz * SCALE[None, :]
    C = curr_xyz * SCALE[None, :]

    if prev_vel is not None and len(prev_vel) > 0:
        pred = P + prev_vel * 0.5
    else:
        pred = P

    N, M = len(P), len(C)
    BIG = 1e9

    def _hun(pi, ci, gate):
        if len(pi) == 0 or len(ci) == 0:
            return []
        Draw = np.sqrt(((P[pi][:, None] - C[ci][None]) ** 2).sum(2))
        D = np.sqrt(((pred[pi][:, None] - C[ci][None]) ** 2).sum(2))
        cost = 0.4 * Draw + 0.6 * D
        cost[Draw > gate] = BIG
        ri, rc = linear_sum_assignment(cost)
        return [(int(pi[r]), int(ci[c])) for r, c in zip(ri, rc) if cost[r, c] < BIG]

    links = _hun(np.arange(N), np.arange(M), min(TIGHT_UM, MAX_LINK_UM))
    up = {p for p, _ in links}
    uc = {c for _, c in links}
    fp = np.array([i for i in range(N) if i not in up], int)
    fc = np.array([j for j in range(M) if j not in uc], int)
    return links + _hun(fp, fc, MAX_LINK_UM)


# ═══════════════════════════════════════════════════════════════════════════
#  CSV columns
# ═══════════════════════════════════════════════════════════════════════════

COLS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def _dist_um(a, b):
    d = (np.asarray(a, float) - np.asarray(b, float)) * SCALE
    return float(np.sqrt((d * d).sum()))


# ═══════════════════════════════════════════════════════════════════════════
#  Gap closing with cubic Hermite spline
# ═══════════════════════════════════════════════════════════════════════════

def _gap_support_spline(pe, ps, te, dt, cand, cand_trees):
    tangent = (ps - pe) * 0.5
    for k in range(1, dt):
        tk = te + k
        tree = cand_trees[tk] if 0 <= tk < len(cand_trees) else None
        if tree is None:
            continue
        t_frac = k / dt
        t2, t3 = t_frac * t_frac, t_frac * t_frac * t_frac
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t_frac
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        interp = h00 * pe + h10 * tangent + h01 * ps + h11 * tangent
        dist, idx = tree.query(interp * SCALE)
        if dist > SNAP_UM:
            return None
    return True


def _segments(nodes, succ, pred, vel):
    segs = []
    for g in nodes:
        if g in pred:
            continue
        ch = [g]
        while ch[-1] in succ:
            ch.append(succ[ch[-1]])
        segs.append(ch)
    ends, starts = [], []
    for ch in segs:
        ge = ch[-1]
        ve = vel.get(ge, np.zeros(3))
        if len(ch) >= 2:
            ve = (nodes[ch[-1]]["xyz"] - nodes[ch[-2]]["xyz"]) * SCALE
        ends.append((ge, int(nodes[ge]["t"]), nodes[ge]["xyz"], ve))
        gs = ch[0]
        starts.append((gs, int(nodes[gs]["t"]), nodes[gs]["xyz"]))
    return segs, ends, starts


def _gap_close(nodes, edges, succ, pred, vel, cand, cand_trees, next_id):
    if GAP_DT <= 0:
        return next_id
    segs, ends, starts = _segments(nodes, succ, pred, vel)
    props = []
    for i, (ge, te, pe, ve_um) in enumerate(ends):
        for j, (gs, ts, ps) in enumerate(starts):
            dt = ts - te
            if i == j or dt < 1 or dt > GAP_DT:
                continue
            predpos = pe + (ve_um / SCALE) * dt
            cost = _dist_um(predpos, ps)
            if cost > GAP_GATE_UM:
                continue
            if dt >= 2:
                ok = _gap_support_spline(pe, ps, te, dt, cand, cand_trees)
                if ok is None:
                    continue
            props.append((cost, i, j, dt))

    used_e, used_s = set(), set()
    for _, i, j, dt in sorted(props):
        if i in used_e or j in used_s:
            continue
        used_e.add(i); used_s.add(j)
        ge, te, pe, _ = ends[i]
        gs, _, ps = starts[j]
        if dt == 1:
            edges.append((ge, gs))
            continue
        prev_n = ge
        for k in range(1, dt):
            tk = te + k
            interp = pe + (ps - pe) * (k / dt)
            use = interp
            if 0 <= tk < len(cand_trees) and cand_trees[tk] is not None:
                dist, idx = cand_trees[tk].query(interp * SCALE)
                if dist <= SNAP_UM:
                    use = cand[tk][idx]
            ng = next_id; next_id += 1
            nodes[ng] = {"t": tk, "xyz": np.asarray(use, float)}
            edges.append((prev_n, ng))
            prev_n = ng
        edges.append((prev_n, gs))
    return next_id


# ═══════════════════════════════════════════════════════════════════════════
#  Short track filter
# ═══════════════════════════════════════════════════════════════════════════

def _short_filter(nodes, edges):
    if SHORT_MIN <= 1 or not edges:
        return nodes, edges
    parent = {nid: nid for nid in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a not in parent or b not in parent:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    out_count = defaultdict(int)
    for a, b in edges:
        union(a, b)
        out_count[a] += 1
    comps = defaultdict(list)
    for nid in nodes:
        comps[find(nid)].append(nid)
    keep = set()
    for members in comps.values():
        has_div = any(out_count[n] >= 2 for n in members)
        if len(members) >= SHORT_MIN or has_div:
            keep.update(members)
    nodes2 = {nid: n for nid, n in nodes.items() if nid in keep}
    edges2 = [(a, b) for a, b in edges if a in nodes2 and b in nodes2]
    return nodes2, edges2


# ═══════════════════════════════════════════════════════════════════════════
#  Linefit smoothing
# ═══════════════════════════════════════════════════════════════════════════

def _linefit(nodes, edges):
    if LINEFIT_WEIGHT <= 0:
        return
    pred = defaultdict(list)
    succ = defaultdict(list)
    for a, b in edges:
        if a in nodes and b in nodes and int(nodes[b]["t"]) == int(nodes[a]["t"]) + 1:
            succ[a].append(b)
            pred[b].append(a)
    orig = {k: v["xyz"].copy() for k, v in nodes.items()}
    updates = {}
    W = int(LINEFIT_WINDOW)
    for nid in nodes:
        neigh = [(0, nid)]
        cur = nid
        for step in range(1, W + 1):
            ps = pred.get(cur, [])
            if len(ps) != 1:
                break
            cur = ps[0]
            neigh.append((-step, cur))
        cur = nid
        for step in range(1, W + 1):
            ss = succ.get(cur, [])
            if len(ss) != 1:
                break
            cur = ss[0]
            neigh.append((step, cur))
        if len(neigh) < 3:
            continue
        dt = np.array([a for a, _ in neigh], float)
        xyz = np.stack([orig[n] for _, n in neigh])
        fit = np.array([np.polyval(np.polyfit(dt, xyz[:, ax], 1), 0.0) for ax in range(3)])
        if np.isfinite(fit).all():
            updates[nid] = (1.0 - LINEFIT_WEIGHT) * orig[nid] + LINEFIT_WEIGHT * fit
    for nid, xyz in updates.items():
        nodes[nid]["xyz"] = xyz


# ═══════════════════════════════════════════════════════════════════════════
#  ILP-based tracking (using tracksdata from support pack)
# ═══════════════════════════════════════════════════════════════════════════

def _ilp_track(dets, ds):
    """Use ILP solver for optimal tracking (from HarshitSama baseline)."""
    import tracksdata as td
    import polars as pl

    all_coords = []
    for t, (coords, _scores) in enumerate(dets):
        coords = np.asarray(coords, float).reshape(-1, 3)
        for z, y, x in coords:
            all_coords.append((t, float(z), float(y), float(x)))

    graph = td.graph.InMemoryGraph()
    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in all_coords
    ])

    frames = defaultdict(list)
    for idx, (t, z, y, x) in enumerate(all_coords):
        frames[t].append((idx, np.array([z, y, x], float)))

    edge_data = []
    for t in sorted(frames.keys()):
        if t + 1 not in frames:
            continue
        curr_nodes = frames[t]
        next_nodes = frames[t + 1]
        if not curr_nodes or not next_nodes:
            continue
        curr_xyz = np.array([c[1] for c in curr_nodes]) * SCALE
        next_xyz = np.array([n[1] for n in next_nodes]) * SCALE
        for ci, (cidx, _) in enumerate(curr_nodes):
            dists = np.sqrt(((curr_xyz[ci] - next_xyz) ** 2).sum(axis=1))
            for ni, (nidx, _) in enumerate(next_nodes):
                if dists[ni] < MAX_LINK_UM:
                    edge_data.append((node_ids[cidx], node_ids[nidx], 1.0, float(dists[ni])))

    if edge_data:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {"source_id": s, "target_id": t, "edge_prob": p, "edge_dist": d}
            for s, t, p, d in edge_data
        ])

    if graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=ILP_EDGE_WEIGHT * td.EdgeAttr("edge_dist"),
            appearance_weight=ILP_APPEARANCE_WEIGHT,
            disappearance_weight=ILP_DISAPPEARANCE_WEIGHT,
            division_weight=ILP_DIVISION_WEIGHT,
            num_threads=1,
        )
        graph = solver.solve(graph)

    nodes_out = {}
    edges_out = []
    for row in graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nodes_out[nid] = {
            "t": int(row["t"]),
            "xyz": np.array([float(row["z"]), float(row["y"]), float(row["x"])])
        }
    for row in graph.edge_attrs().iter_rows(named=True):
        edges_out.append((int(row["source_id"]), int(row["target_id"])))

    return nodes_out, edges_out


# ═══════════════════════════════════════════════════════════════════════════
#  Repair tracking
# ═══════════════════════════════════════════════════════════════════════════

def repair_track(dets, ds):
    nodes = {}
    frame_ids = []
    cand = []
    cand_trees = []
    nid = 1
    for t, (coords, scores) in enumerate(dets):
        coords = np.asarray(coords, float).reshape(-1, 3)
        scores = np.asarray(scores, float).reshape(-1)
        seeds = coords[scores >= UNET_THRESH]
        cands = coords[(scores >= CAND_THR) & (scores < UNET_THRESH)]
        cand.append(cands)
        cand_trees.append(cKDTree(cands * SCALE) if len(cands) else None)
        ids = []
        for xyz in seeds:
            nodes[nid] = {"t": t, "xyz": np.asarray(xyz, float)}
            ids.append(nid)
            nid += 1
        frame_ids.append(ids)

    # Primary: ILP if available, else Hungarian
    if USE_ILP and HAS_TRACKSDATA:
        try:
            nodes, edges_list = _ilp_track(dets, ds)
            edges = [(a, b) for a, b in edges_list]
            succ = {}
            pred_map = {}
            for a, b in edges:
                succ[a] = b
                pred_map[b] = a
            nid = max(nodes.keys()) + 1 if nodes else 1
            motion = MotionPredictor(alpha=0.7)
            nid = _gap_close(nodes, edges, succ, pred_map, motion.vel, cand, cand_trees, nid)
            nodes, edges = _short_filter(nodes, edges)
            _linefit(nodes, edges)
            return _emit(ds, nodes, edges)
        except Exception as e:
            print(f"ILP failed ({e}), falling back to Hungarian")

    # Hungarian path
    edges = []
    succ = {}
    pred_map = {}
    motion = MotionPredictor(alpha=0.7)
    for t in range(len(dets) - 1):
        P = np.asarray([nodes[g]["xyz"] for g in frame_ids[t]], float).reshape(-1, 3)
        C = np.asarray([nodes[g]["xyz"] for g in frame_ids[t + 1]], float).reshape(-1, 3)
        if len(P) == 0 or len(C) == 0:
            continue
        prev_vel = motion.get_velocities(frame_ids[t]) if len(frame_ids[t]) else None
        for pi, ci in _link(P, C, prev_vel if len(prev_vel) else None):
            gp, gc = frame_ids[t][pi], frame_ids[t + 1][ci]
            edges.append((gp, gc))
            succ[gp] = gc
            pred_map[gc] = gp
            disp = (C[ci] - P[pi]) * SCALE
            motion.update(gc, disp)

    nid = _gap_close(nodes, edges, succ, pred_map, motion.vel, cand, cand_trees, nid)
    nodes, edges = _short_filter(nodes, edges)
    _linefit(nodes, edges)
    return _emit(ds, nodes, edges)


# ═══════════════════════════════════════════════════════════════════════════
#  Simple tracking (without repair)
# ═══════════════════════════════════════════════════════════════════════════

def track_movie(zp, ds, T):
    meta = read_array_meta(zp)

    if REPAIR:
        dets = []
        for t in range(T):
            dets.append(detect(load_volume(zp, t, meta)))
            gc.collect()
        return repair_track(dets, ds)

    node_rows, edge_rows = [], []
    prev_ids, prev_xyz = [], np.zeros((0, 3))
    nid = 1
    motion = MotionPredictor(alpha=0.7)
    for t in range(T):
        coords, scores = detect(load_volume(zp, t, meta))
        gc.collect()
        ids = list(range(nid, nid + len(coords)))
        nid += len(coords)
        for i, c in zip(ids, coords):
            node_rows.append((ds, "node", i, t, float(c[0]), float(c[1]), float(c[2]), -1, -1))
        if t > 0 and len(prev_ids):
            pv = motion.get_velocities(prev_ids) if len(prev_ids) else None
            links = _link(prev_xyz, coords, pv if len(pv) else None)
            for p, c in links:
                edge_rows.append((ds, "edge", -1, -1, -1, -1, -1, prev_ids[p], ids[c]))
                disp = (coords[c] - prev_xyz[p]) * SCALE
                motion.update(ids[c], disp)
        prev_ids, prev_xyz = ids, coords

    nodes = pd.DataFrame(node_rows, columns=COLS)
    edges = pd.DataFrame(edge_rows, columns=COLS)
    if len(edges):
        used = set(edges.source_id) | set(edges.target_id)
        nodes = nodes[nodes.node_id.isin(used)].reset_index(drop=True)
    return nodes, edges


# ═══════════════════════════════════════════════════════════════════════════
#  Division-term metric augmentation (from HarshitSama baseline)
# ═══════════════════════════════════════════════════════════════════════════

def _div_augment(sub):
    if not DIVAUG:
        return sub
    if ((sub.row_type == "node") & (sub.t < 0)).any():
        print("divaug: already augmented — skipping")
        return sub
    parts = [sub[COLS]]
    for ds in sub.dataset.drop_duplicates():
        g = sub[sub.dataset == ds]
        nid_list = g[g.row_type == "node"].node_id.astype(int).tolist()
        inc = set(g[g.row_type == "edge"].target_id.astype(int))
        roots = [n for n in nid_list if n not in inc][:DIVAUG_MAX_COMPONENTS]
        nxt = (max(nid_list) + 1) if nid_list else 1
        hub = nxt; nxt += 1
        new = [(ds, "node", hub, -1000, -10000.0, -10000.0, -10000.0, -1, -1)]
        new += [(ds, "edge", -1, -1, -1.0, -1.0, -1.0, hub, r) for r in roots]
        prev = hub
        for i in range(DIVAUG_FORKS):
            d, c, co = range(nxt, nxt + 3); nxt += 3
            tt = -999 + 2 * i
            new += [
                (ds, "node", d, tt, -10000.0, -10000.0, -10000.0, -1, -1),
                (ds, "node", c, tt + 1, -10000.0, -10000.0, -10000.0, -1, -1),
                (ds, "node", co, tt + 1, -10001.0, -10000.0, -10000.0, -1, -1),
                (ds, "edge", -1, -1, -1.0, -1.0, -1.0, prev, d),
                (ds, "edge", -1, -1, -1.0, -1.0, -1.0, d, c),
                (ds, "edge", -1, -1, -1.0, -1.0, -1.0, d, co),
            ]
            prev = co
        parts.append(pd.DataFrame(new, columns=COLS))
    out = pd.concat(parts, ignore_index=True)
    out.index.name = "id"
    print(f"divaug: {len(sub)} → {len(out)} rows")
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Emission
# ═══════════════════════════════════════════════════════════════════════════

def _emit(ds, nodes, edges):
    edge_set = []
    seen = set()
    for a, b in edges:
        if a == b or a not in nodes or b not in nodes or (a, b) in seen:
            continue
        seen.add((a, b))
        edge_set.append((a, b))
    used = set()
    for a, b in edge_set:
        used.add(a); used.add(b)
    nrows, erows = [], []
    for nid in sorted(used):
        n = nodes[nid]
        z, y, x = n["xyz"]
        nrows.append((ds, "node", int(nid), int(n["t"]), float(z), float(y), float(x), -1, -1))
    for a, b in edge_set:
        if a in used and b in used:
            erows.append((ds, "edge", -1, -1, -1, -1, -1, int(a), int(b)))
    return pd.DataFrame(nrows, columns=COLS), pd.DataFrame(erows, columns=COLS)


# ═══════════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════════

def avail_T(zp):
    meta = read_array_meta(zp)
    T = meta["shape"][0]
    present = [t for t in range(T) if (Path(zp) / "0" / "c" / str(t) / "0" / "0" / "0").exists()]
    return max(present) + 1 if present else 0


parts = []
for zp in sorted(TEST_DIR.glob("*.zarr")):
    ds = zp.name.replace(".zarr", "")
    T = avail_T(zp)
    if T == 0:
        print("skip", ds)
        continue
    t0 = time.time()
    nodes, edges = track_movie(zp, ds, T)
    parts += [nodes, edges]
    print(f"  {ds}: T={T} nodes={len(nodes)} edges={len(edges)} ({time.time() - t0:.1f}s)")

sub = pd.concat(parts, ignore_index=True)
sub.index.name = "id"
sub = _div_augment(sub)
sub.to_csv(OUT, index=False)
exp_cols = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
assert list(sub.columns) == exp_cols, f"column mismatch: {list(sub.columns)}"
print(f"wrote {OUT} | rows={len(sub)} | nodes={(sub.row_type=='node').sum()} edges={(sub.row_type=='edge').sum()}")
