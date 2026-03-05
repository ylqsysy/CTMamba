#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "configs").exists() and (p / "src").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]


repo_root = _repo_root()

import sys  # noqa: E402

for extra in (repo_root, repo_root / "src"):
    extra_s = str(extra)
    if extra_s not in sys.path:
        sys.path.insert(0, extra_s)

from hsi3d.data.hsi_dataset import HSIPatchDataset, compute_train_norm  # noqa: E402
from hsi3d.utils.io import ensure_dir, load_json, load_yaml, save_json  # noqa: E402
from hsi3d.utils.seed import set_global_seed  # noqa: E402


VALID_BASELINES = ("morphformer", "mambahsi", "igroupss_mamba")


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_disjoint(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    inter = np.intersect1d(a, b, assume_unique=False)
    if inter.size:
        raise ValueError(f"split overlap: {name_a} ∩ {name_b} size={int(inter.size)}")


def _cm_update(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> None:
    k = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    binc = np.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    cm += binc


def _cm_scores(cm: np.ndarray) -> Tuple[float, float, float]:
    cm = cm.astype(np.float64)
    total = cm.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    po = float(np.trace(cm) / total)
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = float((row * col).sum() / (total * total + 1e-12))
    kappa = float((po - pe) / (1.0 - pe + 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_i = np.diag(cm) / (row + 1e-12)
    aa = float(np.nanmean(acc_i))
    oa = float(po)
    return oa, aa, kappa


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module | None = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    total_loss = 0.0
    n_seen = 0

    for x, _x_spec, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)

        pred = logits.argmax(dim=1).detach().cpu().numpy()
        yt = y.detach().cpu().numpy()
        _cm_update(cm, yt, pred, num_classes)

        if criterion is not None:
            loss = criterion(logits.float(), y)
            bs = int(x.shape[0])
            total_loss += float(loss.detach().item()) * bs
            n_seen += bs

    oa, aa, kappa = _cm_scores(cm)
    out: Dict[str, float] = {"OA": oa, "AA": aa, "Kappa": kappa}
    if criterion is not None and n_seen > 0:
        out["loss"] = float(total_loss / n_seen)
    return out


def _get(d: Dict[str, Any], keys: Tuple[str, ...], default: Any) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()


def _load_mat(path: Path, key: str, want: str) -> np.ndarray:
    key = str(key or "").strip()

    try:
        import scipy.io  # type: ignore
    except Exception:
        scipy = None  # type: ignore
    else:
        scipy = scipy.io  # type: ignore

    if scipy is not None:
        d = scipy.loadmat(str(path))
        d = {k: v for k, v in d.items() if not k.startswith("__")}
        if key:
            if key not in d:
                raise KeyError(f"key '{key}' not found in {path.name}")
            return np.asarray(d[key])
        cand = []
        for k, v in d.items():
            if not isinstance(v, np.ndarray) or v.size <= 0:
                continue
            if want == "cube" and v.ndim >= 3:
                cand.append((v.size, k, v))
            if want == "gt" and v.ndim >= 2:
                cand.append((v.size, k, v))
        if not cand:
            raise KeyError(f"no suitable array in {path.name}")
        cand.sort(key=lambda x: x[0], reverse=True)
        return np.asarray(cand[0][2])

    import h5py  # type: ignore

    with h5py.File(str(path), "r") as f:
        if key:
            if key not in f:
                raise KeyError(f"key '{key}' not found in {path.name}")
            return np.array(f[key])
        cand = []
        for k in f.keys():
            v = f[k]
            if not hasattr(v, "shape"):
                continue
            shape = tuple(int(x) for x in v.shape)
            size = int(np.prod(shape)) if shape else 0
            if size <= 0:
                continue
            if want == "cube" and len(shape) >= 3:
                cand.append((size, k))
            if want == "gt" and len(shape) >= 2:
                cand.append((size, k))
        if not cand:
            raise KeyError(f"no suitable dataset in {path.name}")
        cand.sort(key=lambda x: x[0], reverse=True)
        return np.array(f[cand[0][1]])


def _ensure_hwc(cube: np.ndarray) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got shape={cube.shape}")
    sh = list(cube.shape)
    bdim = int(np.argmin(sh))
    if bdim == 2:
        return cube
    if bdim == 0:
        return np.transpose(cube, (1, 2, 0))
    return np.transpose(cube, (0, 2, 1))


def _ensure_hw(gt: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    gt = np.asarray(gt)
    if gt.ndim == 3 and 1 in gt.shape:
        gt = np.squeeze(gt)
    if gt.ndim != 2:
        raise ValueError(f"gt must be 2D, got shape={gt.shape}")
    h, w = hw
    if gt.shape == (h, w):
        return gt
    if gt.shape == (w, h):
        return gt.T
    raise ValueError(f"gt shape {gt.shape} does not match cube hw {(h, w)}")


def _load_from_dataset_cfg(dataset: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str, int, int]:
    dataset_name = str(dataset.get("name", dataset.get("dataset", "dataset"))).strip()
    label_offset = int(dataset.get("label_offset", 1))
    num_classes = int(dataset.get("num_classes", dataset.get("n_classes", 0)))
    if num_classes <= 0:
        raise ValueError("num_classes must be provided in dataset config")

    raw_dir = str(dataset.get("raw_dir", "")).strip()
    if not raw_dir:
        raise KeyError("raw_dir missing in dataset config")
    cube_file = str(dataset.get("cube_file", "")).strip()
    gt_file = str(dataset.get("gt_file", "")).strip()
    if not cube_file or not gt_file:
        raise KeyError("cube_file/gt_file missing in dataset config")

    cube_key = str(dataset.get("cube_key", "") or "").strip()
    gt_key = str(dataset.get("gt_key", "") or "").strip()

    cube_path = _resolve_path(Path(raw_dir) / cube_file)
    gt_path = _resolve_path(Path(raw_dir) / gt_file)

    if not cube_path.exists():
        raise FileNotFoundError(f"cube file not found: {cube_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"gt file not found: {gt_path}")

    if cube_path.suffix.lower() in [".npy", ".npz"]:
        cube = np.load(str(cube_path))
        if isinstance(cube, np.lib.npyio.NpzFile):
            raise ValueError(f"cube npz not supported: {cube_path}")
    else:
        cube = _load_mat(cube_path, cube_key, want="cube")

    if gt_path.suffix.lower() in [".npy", ".npz"]:
        gt = np.load(str(gt_path))
        if isinstance(gt, np.lib.npyio.NpzFile):
            raise ValueError(f"gt npz not supported: {gt_path}")
    else:
        gt = _load_mat(gt_path, gt_key, want="gt")

    cube = _ensure_hwc(cube).astype(np.float32, copy=False)
    gt = _ensure_hw(gt, (int(cube.shape[0]), int(cube.shape[1]))).astype(np.int64, copy=False)

    return cube, gt, dataset_name, label_offset, num_classes


def _flat_indices(indices: np.ndarray, H: int, W: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim == 2 and idx.shape[1] == 2:
        idx = (idx[:, 0] * W + idx[:, 1]).astype(np.int64, copy=False)
    else:
        idx = idx.reshape(-1).astype(np.int64, copy=False)
    if idx.size == 0:
        raise ValueError("empty indices")
    mn, mx = int(idx.min()), int(idx.max())
    if mn < 0 or mx >= H * W:
        raise ValueError(f"indices out of range: min={mn}, max={mx}, valid=[0,{H*W-1}]")
    return idx


def _fit_pca_train_only(
    cube_hwb: np.ndarray,
    train_idx_flat: np.ndarray,
    *,
    n_components: int,
    whiten: bool = False,
    random_state: int = 0,
) -> Any:
    try:
        from sklearn.decomposition import PCA  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required when pca_bands > 0. Install it with: pip install scikit-learn"
        ) from e

    H, W, B = cube_hwb.shape
    flat = cube_hwb.reshape(-1, B).astype(np.float64, copy=False)
    Xtr = flat[train_idx_flat]
    pca = PCA(
        n_components=int(n_components),
        whiten=bool(whiten),
        svd_solver="randomized",
        random_state=int(random_state),
    )
    pca.fit(Xtr)
    return pca


def _apply_pca(cube_hwb: np.ndarray, pca: Any) -> np.ndarray:
    H, W, B = cube_hwb.shape
    flat = cube_hwb.reshape(-1, B).astype(np.float64, copy=False)
    Z = pca.transform(flat).astype(np.float32, copy=False)
    return Z.reshape(H, W, -1)


def _compute_class_weights_from_train(
    gt: np.ndarray,
    train_indices: np.ndarray,
    label_offset: int,
    num_classes: int,
) -> np.ndarray:
    y = gt.reshape(-1)[train_indices].astype(np.int64) - int(label_offset)
    y = y[(y >= 0) & (y < int(num_classes))]
    cnt = np.bincount(y, minlength=int(num_classes)).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    w = cnt.sum() / (float(num_classes) * cnt)
    w = w / w.mean()
    return w.astype(np.float32)


def _resolve_baseline_cfg(dataset_name: str) -> Path:
    key = "pu" if dataset_name == "pavia_university" else dataset_name
    return repo_root / "configs" / "baselines" / f"{key}.yaml"


class _ConvStem(nn.Module):
    def __init__(self, in_bands: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_bands, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=max(1, embed_dim // 8), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout2d(p=float(max(0.0, dropout))),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MambaLikeBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.0, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        k = int(max(1, kernel_size))
        if k % 2 == 0:
            k += 1
        self.norm1 = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2, groups=dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(float(max(0.0, dropout)))
        self.norm2 = nn.LayerNorm(dim)
        hidden = max(4, int(round(float(mlp_ratio) * dim)))
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hidden, dim),
            nn.Dropout(float(max(0.0, dropout))),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        u, v = self.in_proj(h).chunk(2, dim=-1)
        u = self.dw_conv(u.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop(self.out_proj(F.silu(u) * torch.sigmoid(v)))
        x = x + self.ffn(self.norm2(x))
        return x


class MorphFormerLite(nn.Module):
    def __init__(
        self,
        in_bands: int,
        num_classes: int,
        patch_size: int,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.patch_size = int(patch_size)
        self.stem = _ConvStem(in_bands, embed_dim, dropout=dropout * 0.5)
        token_n = self.patch_size * self.patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, token_n + 1, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=max(4, int(round(float(mlp_ratio) * embed_dim))),
            dropout=float(max(0.0, dropout)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(depth)))
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, int(num_classes))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        tok = feat.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tok.shape[0], -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        if tok.shape[1] == self.pos_embed.shape[1]:
            tok = tok + self.pos_embed
        tok = self.encoder(tok)
        out = self.norm(tok[:, 0, :])
        return self.head(out)


class MambaHSILite(nn.Module):
    def __init__(
        self,
        in_bands: int,
        num_classes: int,
        embed_dim: int = 128,
        depth: int = 6,
        kernel_size: int = 5,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = _ConvStem(in_bands, embed_dim, dropout=dropout * 0.3)
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.blocks = nn.ModuleList(
            [
                _MambaLikeBlock(
                    dim=embed_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(max(1, int(depth)))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = feat * self.spectral_gate(feat)
        tok = feat.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            tok = blk(tok)
        out = self.norm(tok).mean(dim=1)
        return self.head(out)


class IGroupSSMambaLite(nn.Module):
    def __init__(
        self,
        in_bands: int,
        num_classes: int,
        embed_dim: int = 160,
        depth: int = 8,
        kernel_size: int = 5,
        num_groups: int = 25,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_groups = max(1, int(num_groups))
        self.stem = _ConvStem(in_bands, embed_dim, dropout=dropout * 0.25)
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList(
            [
                _MambaLikeBlock(
                    dim=embed_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(max(1, int(depth)))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, int(num_classes))

    def _group_tokens(self, tok: torch.Tensor) -> torch.Tensor:
        b, n, c = tok.shape
        g = min(self.num_groups, n)
        if n % g != 0:
            pad = g - (n % g)
            tok = F.pad(tok, (0, 0, 0, pad))
            n = tok.shape[1]
        chunk = n // g
        return tok.view(b, g, chunk, c).mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        tok = feat.flatten(2).transpose(1, 2)
        tok = self._group_tokens(tok)
        tok = self.pre_norm(tok)
        for blk in self.blocks:
            tok = blk(tok)
        out = self.norm(tok).mean(dim=1)
        return self.head(out)


def _build_model(
    baseline: str,
    cfg: Dict[str, Any],
    *,
    bands: int,
    num_classes: int,
    patch_size: int,
) -> nn.Module:
    if baseline == "morphformer":
        return MorphFormerLite(
            in_bands=bands,
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dim=int(cfg.get("embed_dim", 96)),
            depth=int(cfg.get("depth", 4)),
            num_heads=int(cfg.get("num_heads", 4)),
            mlp_ratio=float(cfg.get("mlp_ratio", 2.0)),
            dropout=float(cfg.get("dropout", 0.12)),
        )
    if baseline == "mambahsi":
        return MambaHSILite(
            in_bands=bands,
            num_classes=num_classes,
            embed_dim=int(cfg.get("embed_dim", 128)),
            depth=int(cfg.get("depth", 6)),
            kernel_size=int(cfg.get("state_kernel", 5)),
            mlp_ratio=float(cfg.get("mlp_ratio", 2.0)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    if baseline == "igroupss_mamba":
        return IGroupSSMambaLite(
            in_bands=bands,
            num_classes=num_classes,
            embed_dim=int(cfg.get("embed_dim", 160)),
            depth=int(cfg.get("depth", 8)),
            kernel_size=int(cfg.get("state_kernel", 5)),
            num_groups=int(cfg.get("num_groups", 25)),
            mlp_ratio=float(cfg.get("mlp_ratio", 2.0)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"unsupported baseline: {baseline}")


def _select_score(d: Dict[str, float], metric: str) -> float:
    m = str(metric).lower().strip()
    if m in ("kappa", "k"):
        return float(d.get("Kappa", 0.0))
    if m in ("oa",):
        return float(d.get("OA", 0.0))
    if m in ("aa",):
        return float(d.get("AA", 0.0))
    if m in ("loss", "val_loss"):
        return -float(d.get("loss", 1e18))
    return float(d.get("Kappa", 0.0))


def run(default_baseline: str) -> None:
    if default_baseline not in VALID_BASELINES:
        raise ValueError(f"default_baseline must be one of {VALID_BASELINES}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--train_cfg", required=True)  # compatibility; not used
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)

    ap.add_argument("--baseline", default=default_baseline, choices=list(VALID_BASELINES))
    ap.add_argument("--patch_size", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_amp", action="store_true")
    args = ap.parse_args()

    baseline = str(args.baseline).lower().strip()
    if baseline != default_baseline:
        raise ValueError(f"script entry expects baseline='{default_baseline}', got '{baseline}'")

    set_global_seed(int(args.seed))

    dataset = load_yaml(args.dataset_cfg)
    split = load_json(args.split_json)

    cube, gt, dataset_name, label_offset, num_classes = _load_from_dataset_cfg(dataset)
    H, W = int(gt.shape[0]), int(gt.shape[1])

    train_indices = np.unique(_flat_indices(np.asarray(split["train_indices"]), H, W).astype(np.int64, copy=False))
    val_indices = np.unique(_flat_indices(np.asarray(split["val_indices"]), H, W).astype(np.int64, copy=False))
    test_indices = np.unique(_flat_indices(np.asarray(split["test_indices"]), H, W).astype(np.int64, copy=False))

    _assert_disjoint(train_indices, val_indices, "train", "val")
    _assert_disjoint(train_indices, test_indices, "train", "test")
    _assert_disjoint(val_indices, test_indices, "val", "test")

    bcfg_path = _resolve_baseline_cfg(dataset_name)
    if not bcfg_path.exists():
        raise FileNotFoundError(f"baseline yaml not found: {bcfg_path}")
    bcfg = load_yaml(str(bcfg_path))
    baseline_cfg_sha1 = _sha1_file(bcfg_path)

    base_cfg = _get(bcfg, ("baseline", baseline), None)
    if base_cfg is None and isinstance(bcfg, dict):
        base_cfg = bcfg.get(baseline)
    if not isinstance(base_cfg, dict):
        raise KeyError(f"Cannot find baseline.{baseline} in {bcfg_path}")

    def _hp(name: str, default: Any) -> Any:
        return _get(base_cfg, (name,), default)

    yaml_patch_size = int(_hp("patch_size", 15))
    if yaml_patch_size != 15:
        raise ValueError(f"baseline.{baseline}.patch_size must be 15")
    if int(args.patch_size) != yaml_patch_size:
        raise ValueError(f"CLI patch_size={args.patch_size} != yaml patch_size={yaml_patch_size}")
    patch_size = yaml_patch_size

    pca_bands = int(_hp("pca_bands", 0))
    pca_whiten = bool(_hp("pca_whiten", False))

    batch_size = int(_hp("batch_size", 64))
    eval_batch_size = int(_hp("eval_batch_size", 256))
    num_workers = int(_hp("num_workers", 0))
    pin_memory = bool(_hp("pin_memory", True))

    lr = float(_hp("lr", 5e-4))
    weight_decay = float(_hp("weight_decay", 1e-2))
    betas = _hp("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError(f"baseline.{baseline}.betas must be a list of length 2")
    betas_tuple = (float(betas[0]), float(betas[1]))
    grad_clip = float(_hp("grad_clip", 1.0))

    max_epochs = int(_hp("max_epochs", 160))
    early_stop_patience = int(_hp("early_stop_patience", 20))
    min_epochs = int(_hp("min_epochs", 40))
    select_metric = str(_hp("select_metric", "kappa")).lower().strip()

    scheduler_name = str(_hp("scheduler", "cosine")).lower().strip()
    min_lr = float(_hp("min_lr", 1e-6))

    augment = bool(_hp("augment", False))
    label_smoothing = float(_hp("label_smoothing", 0.0))
    spectral_dropout = float(_hp("spectral_dropout", 0.0))
    input_noise_std = float(_hp("input_noise_std", 0.0))

    class_weight = str(_hp("class_weight", "null")).lower().strip()
    class_weight_power = float(_hp("class_weight_power", 0.0))

    norm_mean_global_blend = float(_hp("norm_mean_global_blend", 0.0))
    norm_std_global_ratio = float(_hp("norm_std_global_ratio", 0.05))
    norm_std_abs_floor = float(_hp("norm_std_abs_floor", 1e-3))

    cfg_use_amp = bool(_hp("use_amp", False))
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = bool((args.use_amp or cfg_use_amp) and device.type == "cuda")

    if select_metric not in {"kappa", "oa", "aa", "loss"}:
        raise ValueError("select_metric must be one of: kappa / oa / aa / loss")
    if pca_bands < 0:
        raise ValueError("pca_bands must be >= 0")

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(str(ckpt_dir))
    meta_dir = out_dir / "meta"
    ensure_dir(str(meta_dir))

    raw_bands = int(cube.shape[-1])
    cube_used = cube.astype(np.float32, copy=False)
    spectral_method = "raw"
    spectral_meta: Dict[str, Any] = {}
    pca_path = meta_dir / "pca_params.npz"
    if pca_bands > 0:
        if pca_bands > raw_bands:
            raise ValueError(f"pca_bands={pca_bands} exceeds raw bands={raw_bands}")
        spectral_method = "pca"
        if pca_path.exists():
            npz = np.load(str(pca_path))
            pca_mean = npz["mean"].astype(np.float32)
            pca_comps = npz["comps"].astype(np.float32)
            spectral_meta = {"pca_k": int(pca_comps.shape[0]), "pca_whiten": bool(pca_whiten), "solver": "cached"}
        else:
            pca = _fit_pca_train_only(
                cube_hwb=cube_used,
                train_idx_flat=train_indices,
                n_components=pca_bands,
                whiten=pca_whiten,
                random_state=int(args.seed),
            )
            pca_mean = pca.mean_.astype(np.float32, copy=False)
            pca_comps = pca.components_.astype(np.float32, copy=False)
            np.savez_compressed(pca_path, mean=pca_mean, comps=pca_comps)
            spectral_meta = {"pca_k": int(pca_bands), "pca_whiten": bool(pca_whiten), "solver": "randomized"}

        class _SimplePCA:
            def __init__(self, mean: np.ndarray, comps: np.ndarray) -> None:
                self.mean = mean
                self.comps = comps

            def transform(self, x: np.ndarray) -> np.ndarray:
                x = x.astype(np.float32, copy=False)
                return (x - self.mean.reshape(1, -1)) @ self.comps.T

        cube_used = _apply_pca(cube_used, _SimplePCA(pca_mean, pca_comps))

    bands = int(cube_used.shape[-1])

    norm_path = meta_dir / "norm_stats.npz"
    if norm_path.exists():
        npz = np.load(str(norm_path))
        mean = npz["mean"].astype(np.float32)
        std = npz["std"].astype(np.float32)
    else:
        mean, std = compute_train_norm(
            cube_used,
            train_indices,
            mean_global_blend=norm_mean_global_blend,
            std_global_ratio=norm_std_global_ratio,
            std_abs_floor=norm_std_abs_floor,
        )
        np.savez_compressed(norm_path, mean=mean, std=std)

    g = torch.Generator()
    g.manual_seed(int(args.seed))

    ds_tr = HSIPatchDataset(
        cube=cube_used,
        gt=gt,
        indices=train_indices,
        patch_size=patch_size,
        label_offset=label_offset,
        mean=mean,
        std=std,
        augment=augment,
    )
    ds_va = HSIPatchDataset(
        cube=cube_used,
        gt=gt,
        indices=val_indices,
        patch_size=patch_size,
        label_offset=label_offset,
        mean=mean,
        std=std,
        augment=False,
    )
    ds_te = HSIPatchDataset(
        cube=cube_used,
        gt=gt,
        indices=test_indices,
        patch_size=patch_size,
        label_offset=label_offset,
        mean=mean,
        std=std,
        augment=False,
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = _build_model(baseline, base_cfg, bands=bands, num_classes=num_classes, patch_size=patch_size).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas_tuple)

    scheduler = None
    if scheduler_name in {"cos", "cosine"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs, eta_min=min_lr)
    elif scheduler_name in {"none", "off", ""}:
        scheduler = None
    elif scheduler_name in {"step", "steplr"}:
        step_size = int(_hp("step_size", 40))
        gamma = float(_hp("gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=max(1, step_size), gamma=gamma)
    else:
        raise ValueError(f"unsupported scheduler='{scheduler_name}'")

    cw_tensor = None
    if class_weight in {"balanced", "balance"} and class_weight_power > 0:
        w = _compute_class_weights_from_train(gt, train_indices, label_offset, num_classes)
        alpha = float(np.clip(class_weight_power, 0.0, 1.0))
        w = (1.0 - alpha) * np.ones_like(w, dtype=np.float32) + alpha * w
        cw_tensor = torch.tensor(w, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=label_smoothing)

    spec_dropout = nn.Dropout2d(p=spectral_dropout) if spectral_dropout > 0 else None
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_score = -1e18
    best_ep = -1
    bad = 0
    best_path = ckpt_dir / "best.pt"

    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        ep_loss = 0.0
        n_seen = 0

        for x, _x_spec, y in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                x_in = x
                if spec_dropout is not None:
                    x_in = spec_dropout(x_in)
                if input_noise_std > 0:
                    x_in = x_in + torch.randn_like(x_in) * input_noise_std
                logits = model(x_in)
            loss = criterion(logits.float(), y)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optim)
            scaler.update()

            bs_now = int(x.shape[0])
            ep_loss += float(loss.detach().item()) * bs_now
            n_seen += bs_now

        ep_loss = ep_loss / max(1, n_seen)
        if scheduler is not None:
            scheduler.step()

        val = _evaluate(model, dl_va, device, num_classes, criterion=criterion, use_amp=use_amp)
        score = _select_score(val, select_metric)
        improved = score > best_score + 1e-12
        if improved:
            best_score = score
            best_ep = ep
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": {
                        "dataset": dataset_name,
                        "baseline": baseline,
                        "seed": int(args.seed),
                        "patch_size": int(patch_size),
                        "bands": int(bands),
                        "raw_bands": int(raw_bands),
                        "spectral_method": spectral_method,
                        "spectral_meta": spectral_meta,
                        "num_classes": int(num_classes),
                        "label_offset": int(label_offset),
                        "baseline_cfg": str(bcfg_path),
                        "baseline_cfg_sha1": baseline_cfg_sha1,
                        "select_metric": select_metric,
                    },
                },
                best_path,
            )
        else:
            if ep >= min_epochs:
                bad += 1

        if early_stop_patience > 0 and bad >= early_stop_patience:
            break

        if ep % 10 == 0 or improved:
            dt = time.time() - t0
            print(
                f"[ep {ep:04d}] loss={ep_loss:.6f} "
                f"| VAL OA={val['OA']:.4f} AA={val['AA']:.4f} Kappa={val['Kappa']:.4f} "
                f"| score={score:.6f} best={best_score:.6f}@{best_ep} bad={bad} "
                f"| t={dt:.1f}s | baseline={baseline} spectral={spectral_method} bands={bands}"
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    val_best = _evaluate(model, dl_va, device, num_classes, criterion=criterion, use_amp=use_amp)
    test_best = _evaluate(model, dl_te, device, num_classes, criterion=criterion, use_amp=use_amp)

    metrics = {
        "VAL": val_best,
        "TEST": test_best,
        "meta": {
            "dataset": dataset_name,
            "baseline": baseline,
            "split_json": str(Path(args.split_json)),
            "ckpt": str(best_path),
            "ckpt_key": "model",
            "label_offset": int(label_offset),
            "num_classes": int(num_classes),
            "patch_size": int(patch_size),
            "bands": int(bands),
            "raw_bands": int(raw_bands),
            "spectral_method": spectral_method,
            "spectral_meta": spectral_meta,
            "norm_path": str(norm_path),
            "baseline_cfg": str(bcfg_path),
            "baseline_cfg_sha1": baseline_cfg_sha1,
            "best_ep": int(best_ep),
            "select_metric": select_metric,
            "class_weight": class_weight,
            "class_weight_power": float(class_weight_power),
            "label_smoothing": float(label_smoothing),
            "spectral_dropout": float(spectral_dropout),
            "input_noise_std": float(input_noise_std),
            "augment": bool(augment),
            "scheduler": scheduler_name,
            "use_amp": bool(use_amp),
        },
    }
    save_json(out_dir / "metrics.json", metrics)
    print(f"[done] out_dir={out_dir}")


def main() -> None:
    run(default_baseline="morphformer")


if __name__ == "__main__":
    main()
