#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import math
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

from hsi3d.data.hsi_dataset import HSIPatchDataset  # noqa: E402
from hsi3d.utils.io import ensure_dir, load_json, load_yaml, save_json  # noqa: E402
from hsi3d.utils.seed import set_global_seed  # noqa: E402


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


def _resolve_path(p: str) -> Path:
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

    cube_path = _resolve_path(str(Path(raw_dir) / cube_file))
    gt_path = _resolve_path(str(Path(raw_dir) / gt_file))

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


def _train_mean_std(cube: np.ndarray, train_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = cube.reshape(-1, int(cube.shape[-1])).astype(np.float32, copy=False)
    x = flat[train_indices]
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def _compute_class_weights_from_train(
    gt: np.ndarray,
    train_indices: np.ndarray,
    label_offset: int,
    num_classes: int,
) -> np.ndarray:
    flat = gt.reshape(-1).astype(np.int64, copy=False)
    y = flat[train_indices] - int(label_offset)
    y = y[(y >= 0) & (y < int(num_classes))]
    if y.size == 0:
        return np.ones((num_classes,), dtype=np.float32)

    counts = np.bincount(y, minlength=int(num_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (counts * float(num_classes))
    w = w / (w.mean() + 1e-12)
    return w.astype(np.float32)


def _band_sample_indices(b: int, k: int) -> np.ndarray:
    if k >= b:
        return np.arange(b, dtype=np.int64)
    step = max(1, b // k)
    idx = np.arange(0, b, step, dtype=np.int64)[:k]
    return idx


def _spectral_sample(cube: np.ndarray, k: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    idx = _band_sample_indices(int(cube.shape[-1]), int(k))
    z = cube[:, :, idx].astype(np.float32, copy=False)
    return z, {"sample_k": int(k), "sample_idx": idx.tolist()}


def _fit_pca_train_only(
    cube: np.ndarray,
    train_indices: np.ndarray,
    k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    flat = cube.reshape(-1, int(cube.shape[-1])).astype(np.float32, copy=False)
    x = flat[train_indices]

    try:
        from sklearn.decomposition import PCA  # type: ignore

        pca = PCA(n_components=int(k), svd_solver="randomized", random_state=int(seed))
        pca.fit(x)
        mean = pca.mean_.astype(np.float32)
        comps = pca.components_.astype(np.float32)
        meta = {"pca_k": int(k), "solver": "sklearn_randomized"}
        return mean, comps, meta
    except Exception:
        pass

    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    comps = vt[:k].astype(np.float32)
    mean = x.mean(axis=0).astype(np.float32)
    meta = {"pca_k": int(k), "solver": "numpy_svd"}
    return mean, comps, meta


def _apply_pca_to_cube(cube: np.ndarray, mean: np.ndarray, comps: np.ndarray, chunk: int = 65536) -> np.ndarray:
    h, w, b = cube.shape
    flat = cube.reshape(-1, b).astype(np.float32, copy=False)
    out = np.empty((flat.shape[0], comps.shape[0]), dtype=np.float32)
    mean = mean.reshape(1, -1).astype(np.float32, copy=False)
    wt = comps.astype(np.float32, copy=False).T

    n = flat.shape[0]
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        out[i:j] = (flat[i:j] - mean) @ wt
    return out.reshape(h, w, comps.shape[0]).astype(np.float32, copy=False)


class ECALayer3D(nn.Module):
    def __init__(self, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=int(k_size), padding=(int(k_size) - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -3)).transpose(-1, -3).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class A2S2KResidual(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        *,
        start_block: bool = False,
        end_block: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding)

        self.pre_bn = None if start_block else nn.BatchNorm3d(channels)
        self.bn1 = nn.BatchNorm3d(channels)
        self.bn2 = nn.BatchNorm3d(channels)
        self.attn = ECALayer3D()

        self.start_block = bool(start_block)
        self.end_block = bool(end_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.start_block:
            out = self.conv1(x)
        else:
            assert self.pre_bn is not None
            out = self.pre_bn(x)
            out = F.relu(out, inplace=True)
            out = self.conv1(out)

        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out)

        out = self.attn(out)
        out = out + identity

        if self.end_block:
            out = self.bn2(out)
            out = F.relu(out, inplace=True)

        return out


class A2S2KResNet(nn.Module):
    def __init__(
        self,
        bands: int,
        num_classes: int,
        *,
        kernel_channels: int = 24,
        reduction: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        bands = int(bands)
        if bands < 7:
            raise ValueError(f"A2S2K requires at least 7 bands after spectral reduction, got {bands}")

        kch = int(kernel_channels)
        red = max(1, int(reduction))
        squeeze = max(1, bands // red)
        kernel_3d = max(1, math.ceil((bands - 6) / 2))

        self.conv1x1 = nn.Conv3d(
            in_channels=1,
            out_channels=kch,
            kernel_size=(1, 1, 7),
            stride=(1, 1, 2),
            padding=0,
        )
        self.conv3x3 = nn.Conv3d(
            in_channels=1,
            out_channels=kch,
            kernel_size=(3, 3, 7),
            stride=(1, 1, 2),
            padding=(1, 1, 0),
        )

        self.bn1x1 = nn.Sequential(
            nn.BatchNorm3d(kch, eps=1e-3, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.bn3x3 = nn.Sequential(
            nn.BatchNorm3d(kch, eps=1e-3, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_se = nn.Sequential(
            nn.Conv3d(kch, squeeze, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv_ex = nn.Conv3d(squeeze, kch, kernel_size=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.res_net1 = A2S2KResidual(kch, (1, 1, 7), (0, 0, 3), start_block=True)
        self.res_net2 = A2S2KResidual(kch, (1, 1, 7), (0, 0, 3))
        self.conv2 = nn.Conv3d(
            in_channels=kch,
            out_channels=128,
            kernel_size=(1, 1, kernel_3d),
            padding=0,
            stride=(1, 1, 1),
        )
        self.bn2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=1e-3, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Conv3d(
            in_channels=1,
            out_channels=kch,
            kernel_size=(3, 3, 128),
            padding=0,
            stride=(1, 1, 1),
        )
        self.bn3 = nn.Sequential(
            nn.BatchNorm3d(kch, eps=1e-3, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )

        self.res_net3 = A2S2KResidual(kch, (3, 3, 1), (1, 1, 0))
        self.res_net4 = A2S2KResidual(kch, (3, 3, 1), (1, 1, 0), end_block=True)

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head_dropout = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(kch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"A2S2K expects x as (N,Bands,P,P), got shape={tuple(x.shape)}")

        x = x.permute(0, 2, 3, 1).unsqueeze(1).contiguous()

        x_1x1 = self.bn1x1(self.conv1x1(x)).unsqueeze(dim=1)
        x_3x3 = self.bn3x3(self.conv3x3(x)).unsqueeze(dim=1)

        x_cat = torch.cat([x_3x3, x_1x1], dim=1)
        u = torch.sum(x_cat, dim=1)
        s = self.pool(u)
        z = self.conv_se(s)
        shared_attn = self.conv_ex(z)
        attention = torch.stack([shared_attn, shared_attn], dim=1)
        attention = self.softmax(attention)
        v = (x_cat * attention).sum(dim=1)

        x2 = self.res_net1(v)
        x2 = self.res_net2(x2)
        x2 = self.bn2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1).contiguous()
        x2 = self.bn3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pool(x3).flatten(1)
        x4 = self.head_dropout(x4)
        return self.head(x4)


def _resolve_baseline_cfg(dataset_name: str) -> Path:
    key = "pu" if dataset_name == "pavia_university" else dataset_name
    return repo_root / "configs" / "baselines" / f"{key}.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--train_cfg", required=True)  # compatibility; not used
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)

    ap.add_argument("--baseline", default="a2s2k")
    ap.add_argument("--patch_size", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_amp", action="store_true")
    args = ap.parse_args()

    set_global_seed(int(args.seed))

    dataset = load_yaml(args.dataset_cfg)
    split = load_json(args.split_json)

    cube, gt, dataset_name, label_offset, num_classes = _load_from_dataset_cfg(dataset)

    train_indices = np.unique(np.asarray(split["train_indices"], dtype=np.int64))
    val_indices = np.unique(np.asarray(split["val_indices"], dtype=np.int64))
    test_indices = np.unique(np.asarray(split["test_indices"], dtype=np.int64))

    _assert_disjoint(train_indices, val_indices, "train", "val")
    _assert_disjoint(train_indices, test_indices, "train", "test")
    _assert_disjoint(val_indices, test_indices, "val", "test")

    bcfg_path = _resolve_baseline_cfg(dataset_name)
    if not bcfg_path.exists():
        raise FileNotFoundError(f"A2S2K baseline yaml not found: {bcfg_path}")
    bcfg = load_yaml(str(bcfg_path))
    baseline_cfg_sha1 = _sha1_file(bcfg_path)

    a2s2k_cfg = _get(bcfg, ("baseline", "a2s2k"), None)
    if a2s2k_cfg is None and isinstance(bcfg, dict):
        a2s2k_cfg = bcfg.get("a2s2k")
    if not isinstance(a2s2k_cfg, dict):
        raise KeyError(f"Cannot find baseline.a2s2k in {bcfg_path}")

    def _hp(name: str, default: Any) -> Any:
        return _get(a2s2k_cfg, (name,), default)

    yaml_patch_size = int(_hp("patch_size", 15))
    if yaml_patch_size != 15:
        raise ValueError("baseline.a2s2k.patch_size must be 15")
    if int(args.patch_size) != yaml_patch_size:
        raise ValueError(f"CLI patch_size={args.patch_size} != yaml patch_size={yaml_patch_size}")
    patch_size = yaml_patch_size

    pca_bands = int(_hp("pca_bands", 24))
    kernel_channels = int(_hp("kernel_channels", 24))
    reduction = int(_hp("reduction", 2))
    dropout = float(_hp("dropout", 0.0))

    batch_size = int(_hp("batch_size", 32))
    eval_batch_size = int(_hp("eval_batch_size", 256))
    num_workers = int(_hp("num_workers", 0))

    lr = float(_hp("lr", 6e-4))
    weight_decay = float(_hp("weight_decay", 1e-2))
    betas = _hp("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("baseline.a2s2k.betas must be a list of length 2")
    betas_tuple = (float(betas[0]), float(betas[1]))
    grad_clip = float(_hp("grad_clip", 1.0))

    max_epochs = int(_hp("max_epochs", 160))
    early_stop_patience = int(_hp("early_stop_patience", 20))
    min_epochs = int(_hp("min_epochs", 40))
    select_metric = str(_hp("select_metric", "kappa")).lower().strip()

    augment = bool(_hp("augment", False))
    label_smoothing = float(_hp("label_smoothing", 0.0))
    spectral_dropout = float(_hp("spectral_dropout", 0.0))
    input_noise_std = float(_hp("input_noise_std", 0.0))

    class_weight = str(_hp("class_weight", "null")).lower().strip()
    class_weight_power = float(_hp("class_weight_power", 0.0))

    cfg_use_amp = bool(_hp("use_amp", False))
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = bool((args.use_amp or cfg_use_amp) and device.type == "cuda")

    if select_metric not in {"kappa", "oa", "loss"}:
        raise ValueError("select_metric must be one of: kappa / oa / loss")

    raw_bands = int(cube.shape[-1])
    cube_used = cube.astype(np.float32, copy=False)

    spectral_method = "raw"
    spectral_meta: Dict[str, Any] = {}
    bands = int(cube_used.shape[-1])

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(str(ckpt_dir))
    meta_dir = out_dir / "meta"
    ensure_dir(str(meta_dir))

    if pca_bands != 0:
        if pca_bands > 0:
            spectral_method = "pca"
            pca_path = meta_dir / "pca_params.npz"
            if pca_path.exists():
                npz = np.load(str(pca_path))
                pca_mean = npz["mean"].astype(np.float32)
                pca_comps = npz["comps"].astype(np.float32)
                spectral_meta = {"pca_k": int(pca_comps.shape[0]), "solver": str(npz.get("solver", "cached"))}
            else:
                pca_mean, pca_comps, pmeta = _fit_pca_train_only(cube_used, train_indices, int(pca_bands), int(args.seed))
                spectral_meta = dict(pmeta)
                np.savez_compressed(
                    pca_path,
                    mean=pca_mean,
                    comps=pca_comps,
                    solver=np.array([spectral_meta.get("solver", "")], dtype=object),
                )
            cube_used = _apply_pca_to_cube(cube_used, pca_mean, pca_comps)
            bands = int(cube_used.shape[-1])
        else:
            spectral_method = "sample"
            k = int(abs(pca_bands))
            samp_path = meta_dir / "sample_idx.npz"
            if samp_path.exists():
                npz = np.load(str(samp_path))
                idx = npz["idx"].astype(np.int64)
                cube_used = cube_used[:, :, idx].astype(np.float32, copy=False)
                spectral_meta = {"sample_k": int(idx.size), "sample_idx": idx.tolist()}
                bands = int(cube_used.shape[-1])
            else:
                cube_used, spectral_meta = _spectral_sample(cube_used, k)
                bands = int(cube_used.shape[-1])
                np.savez_compressed(samp_path, idx=np.array(spectral_meta["sample_idx"], dtype=np.int64))

    if bands < 7:
        raise ValueError(f"A2S2K needs at least 7 bands after reduction, got {bands}")

    norm_path = meta_dir / "norm_stats.npz"
    if norm_path.exists():
        npz = np.load(str(norm_path))
        mean = npz["mean"].astype(np.float32)
        std = npz["std"].astype(np.float32)
    else:
        mean, std = _train_mean_std(cube_used, train_indices)
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
        pin_memory=True,
        generator=g,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = A2S2KResNet(
        bands=bands,
        num_classes=num_classes,
        kernel_channels=kernel_channels,
        reduction=reduction,
        dropout=dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas_tuple)

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

        val = _evaluate(model, dl_va, device, num_classes, criterion=criterion, use_amp=use_amp)

        if select_metric == "kappa":
            score = float(val.get("Kappa", 0.0))
        elif select_metric == "oa":
            score = float(val.get("OA", 0.0))
        else:
            score = -float(val.get("loss", 1e18))

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
                        "baseline": "a2s2k",
                        "seed": int(args.seed),
                        "patch_size": int(patch_size),
                        "bands": int(bands),
                        "raw_bands": int(raw_bands),
                        "spectral_method": spectral_method,
                        "spectral_meta": spectral_meta,
                        "num_classes": int(num_classes),
                        "label_offset": int(label_offset),
                        "kernel_channels": int(kernel_channels),
                        "reduction": int(reduction),
                        "dropout": float(dropout),
                        "baseline_cfg": str(bcfg_path),
                        "baseline_cfg_sha1": baseline_cfg_sha1,
                        "select_metric": select_metric,
                        "class_weight": class_weight,
                        "class_weight_power": float(class_weight_power),
                        "label_smoothing": float(label_smoothing),
                        "spectral_dropout": float(spectral_dropout),
                        "input_noise_std": float(input_noise_std),
                        "augment": bool(augment),
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
            vloss = val.get("loss", float("nan"))
            print(
                f"[ep {ep:04d}] loss={ep_loss:.6f} "
                f"| VAL OA={val['OA']:.4f} AA={val['AA']:.4f} Kappa={val['Kappa']:.4f} "
                f"| VAL loss={vloss:.6f} "
                f"| score={score:.6f} best={best_score:.6f}@{best_ep} bad={bad} "
                f"| t={dt:.1f}s "
                f"| spectral={spectral_method} bands={bands}"
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
            "baseline": "a2s2k",
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
            "kernel_channels": int(kernel_channels),
            "reduction": int(reduction),
            "dropout": float(dropout),
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
        },
    }
    save_json(out_dir / "metrics.json", metrics)
    print(f"[done] out_dir={out_dir}")


if __name__ == "__main__":
    main()
