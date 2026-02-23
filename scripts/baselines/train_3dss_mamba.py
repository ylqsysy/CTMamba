#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline runner: 3DSS-Mamba (VideoMamba) integrated into this repo's split protocol.

Key guarantees:
- Uses split_json["train_indices"/"val_indices"/"test_indices"] (NO random split).
- Computes normalization and PCA on TRAIN pixels only (no label leakage).
- Outputs metrics.json with {"val": {...}, "test": {...}} so scripts/run_baselines_10runs_and_mean.py can aggregate.

Expected baseline config location:
  configs/baselines/<dataset>.yaml
with a section:
  baseline:
    3dss_mamba:
      patch_size: 15
      spectral_method: pca
      pca_bands: 30
      ...

The external reference implementation is expected at:
  external/baselines/3dss_mamba
and must expose: models/videomamba.py: VisionMamba
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import time
import ctypes
import site
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "configs").exists() and (p / "src" / "hsi3d").exists():
            return p
        p = p.parent
    # scripts/baselines -> repo_root
    return Path(__file__).resolve().parents[2]


repo_root = _repo_root()

import sys  # noqa: E402

sys.path.insert(0, str(repo_root))

from hsi3d.utils.io import load_yaml, load_json, ensure_dir, save_json  # noqa: E402
from hsi3d.utils.seed import set_global_seed  # noqa: E402
from hsi3d.data.hsi_dataset import HSIPatchDataset, compute_train_norm  # noqa: E402


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()


def _cfg_name(dataset: str) -> str:
    return "pu" if dataset == "pavia_university" else dataset


def _get(d: Dict[str, Any], keys: Tuple[str, ...], default: Any) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _ensure_hwb_cube(cube: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cube = np.asarray(cube)
    gt = np.asarray(gt)
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got shape={cube.shape}")
    if gt.ndim != 2:
        if gt.ndim == 3 and 1 in gt.shape:
            gt = np.squeeze(gt)
        else:
            raise ValueError(f"gt must be 2D, got shape={gt.shape}")

    H, W = gt.shape

    # already (H,W,B)
    if cube.shape[0] == H and cube.shape[1] == W:
        return cube, gt
    # (B,H,W)
    if cube.shape[1] == H and cube.shape[2] == W:
        return np.transpose(cube, (1, 2, 0)), gt
    # (W,H,B)
    if cube.shape[0] == W and cube.shape[1] == H:
        return np.transpose(cube, (1, 0, 2)), gt

    raise ValueError(f"cube/gt mismatch: cube={cube.shape}, gt={gt.shape}")


def _flat_indices(indices: np.ndarray, H: int, W: int) -> np.ndarray:
    idx = np.asarray(indices)
    if idx.ndim == 1:
        return idx.astype(np.int64, copy=False)
    if idx.ndim == 2 and idx.shape[1] == 2:
        r = idx[:, 0].astype(np.int64, copy=False)
        c = idx[:, 1].astype(np.int64, copy=False)
        return (r * W + c).astype(np.int64, copy=False)
    raise ValueError(f"indices must be (N,) or (N,2), got {idx.shape}")


def _fit_pca_train_only(
    cube_hwb: np.ndarray,
    train_flat_idx: np.ndarray,
    n_components: int,
    *,
    whiten: bool,
    random_state: int,
) -> Any:
    from sklearn.decomposition import PCA  # type: ignore

    _H, _W, B = cube_hwb.shape
    flat = cube_hwb.reshape(-1, B).astype(np.float32, copy=False)
    train_flat_idx = np.asarray(train_flat_idx, dtype=np.int64).reshape(-1)
    if train_flat_idx.size == 0:
        raise ValueError("empty train indices for PCA")
    X = flat[train_flat_idx]
    pca = PCA(n_components=int(n_components), whiten=bool(whiten), random_state=int(random_state))
    pca.fit(X)
    return pca


def _apply_pca(cube_hwb: np.ndarray, pca: Any) -> np.ndarray:
    H, W, B = cube_hwb.shape
    flat = cube_hwb.reshape(-1, B).astype(np.float32, copy=False)
    Z = pca.transform(flat).astype(np.float32, copy=False)
    return Z.reshape(H, W, -1)


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


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    use_amp: bool,
    infer_chunk_bs: int = 0,
    phase: str = "eval",
    log_every: int = 0,
    seed: Optional[int] = None,
    max_batches: int = 0,
    prefer_full_batch: bool = True,
) -> Dict[str, float]:
    """Evaluate with the same style as other baselines: one DataLoader batch -> one forward."""
    # Kept for config/call-site compatibility; this implementation intentionally ignores
    # infer_chunk_bs and prefer_full_batch to match other baseline scripts.
    _ = infer_chunk_bs, prefer_full_batch

    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_batches_total = len(dl)
    n_batches = n_batches_total if int(max_batches) <= 0 else min(n_batches_total, int(max_batches))
    t_eval0 = time.time()

    for bi, (x, _x_spec, y) in enumerate(dl, start=1):
        if bi > n_batches:
            break
        x = x.unsqueeze(1).to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)

        pred = logits.argmax(dim=1).detach().cpu().numpy()
        yt = y.detach().cpu().numpy()
        _cm_update(cm, yt, pred, num_classes)

        if log_every and (bi % int(log_every) == 0 or bi == n_batches):
            took = time.time() - t_eval0
            spb = took / max(1, bi)
            eta = spb * (n_batches - bi)
            seed_txt = f"[seed {seed}] " if seed is not None else ""
            print(
                f"[{_now()}] {seed_txt}[{phase}] batch {bi}/{n_batches} "
                f"({100.0 * bi / max(1, n_batches):.1f}%) elapsed={took:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    oa, aa, kappa = _cm_scores(cm)
    return {"OA": oa, "AA": aa, "Kappa": kappa}


def _select_score(m: Dict[str, float], select_metric: str) -> float:
    sel = str(select_metric).lower().strip()
    if sel in ("kappa", "kap"):
        return float(m["Kappa"])
    if sel in ("oa",):
        return float(m["OA"])
    if sel in ("aa",):
        return float(m["AA"])
    if sel in ("min", "min3", "min_oa_aa_kappa"):
        return float(min(m["OA"], m["AA"], m["Kappa"]))
    return float(m["Kappa"])


class _VMambaWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def _enable_speed_flags(*, cudnn_benchmark: bool, tf32: bool) -> None:
    # These flags can significantly speed up training on Ampere+ GPUs (e.g., RTX 4060).
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if tf32 and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def train_one_epoch(
    model: nn.Module,
    dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    grad_clip: float,
    criterion: nn.Module,
    steps_per_epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_seen = 0

    for step, (x, _x_spec, y) in enumerate(dl, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.unsqueeze(1)  # (N,1,B,ps,ps)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits.float(), y)

        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

        scaler.step(optimizer)
        scaler.update()

        bs = int(x.shape[0])
        total_loss += float(loss.detach().item()) * bs
        n_seen += bs

        if steps_per_epoch and step >= steps_per_epoch:
            break

    return float(total_loss / max(1, n_seen))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--train_cfg", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", default="3dss_mamba")
    ap.add_argument("--patch_size", type=int, default=15)
    ap.add_argument("--use_amp", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    seed = int(args.seed)
    set_global_seed(seed)

    out_dir = _resolve_path(args.out_dir)
    ensure_dir(out_dir)
    meta_dir = out_dir / "meta"
    ensure_dir(meta_dir)

    dataset_cfg_path = _resolve_path(args.dataset_cfg)
    train_cfg_path = _resolve_path(args.train_cfg)
    split_path = _resolve_path(args.split_json)

    dataset_cfg = load_yaml(dataset_cfg_path)
    train_cfg = load_yaml(train_cfg_path)
    split = load_json(split_path)

    dataset_name = str(dataset_cfg.get("dataset", dataset_cfg.get("name", "dataset"))).strip()
    label_offset = int(dataset_cfg.get("label_offset", 1))
    num_classes = int(dataset_cfg.get("num_classes", dataset_cfg.get("n_classes", 0)))
    if num_classes <= 0:
        raise ValueError("num_classes missing in dataset config")

    print(
        f"[{_now()}] [seed {seed}] start dataset={dataset_name} "
        f"num_classes={num_classes} patch_size={args.patch_size} out_dir={out_dir}",
        flush=True,
    )

    # ---- baseline cfg ----
    base_tag = _cfg_name(dataset_name)
    baseline_cfg_path = repo_root / "configs" / "baselines" / f"{base_tag}.yaml"
    baseline_all = load_yaml(baseline_cfg_path)
    bcfg = _get(baseline_all, ("baseline", "3dss_mamba"), None)
    if bcfg is None and isinstance(baseline_all, dict) and "3dss_mamba" in baseline_all:
        bcfg = baseline_all["3dss_mamba"]
    if not isinstance(bcfg, dict):
        raise ValueError(f"Cannot find baseline.3dss_mamba in {baseline_cfg_path}")

    patch_size = int(args.patch_size)
    cfg_patch_size = int(bcfg.get("patch_size", patch_size))
    if patch_size != cfg_patch_size:
        raise ValueError(f"patch_size mismatch: cli={patch_size}, cfg={cfg_patch_size}")
    if patch_size != 15:
        raise ValueError("[lock] patch_size must be 15 for paper protocol.")

    # ---- load cube/gt from processed_dir ----
    processed_dir = _resolve_path(str(dataset_cfg.get("processed_dir", "")))
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")

    t_load = time.time()
    cube = np.load(processed_dir / "cube.npy")
    gt = np.load(processed_dir / "gt.npy")
    cube, gt = _ensure_hwb_cube(cube, gt)
    print(
        f"[{_now()}] [seed {seed}] loaded cube={tuple(cube.shape)} gt={tuple(gt.shape)} "
        f"in {time.time() - t_load:.1f}s",
        flush=True,
    )

    H, W, B0 = cube.shape

    train_idx = np.asarray(split.get("train_indices", []))
    val_idx = np.asarray(split.get("val_indices", []))
    test_idx = np.asarray(split.get("test_indices", []))
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"split_json missing indices: {split_path}")

    train_flat = _flat_indices(train_idx, H, W)

    # ---- spectral method ----
    spec_method = str(bcfg.get("spectral_method", "pca")).lower().strip()
    pca_bands = int(bcfg.get("pca_bands", 30))
    if spec_method == "pca":
        if pca_bands < 3:
            raise ValueError("pca_bands must be >= 3 for 3D conv kernel depth.")
        whiten = bool(bcfg.get("pca_whiten", False))

        t_pca = time.time()
        print(
            f"[{_now()}] [seed {seed}] PCA fit start: method=pca bands={pca_bands} "
            f"whiten={whiten} train_pixels={train_flat.size} total_pixels={H*W}",
            flush=True,
        )
        pca = _fit_pca_train_only(cube, train_flat, n_components=pca_bands, whiten=whiten, random_state=seed)
        print(f"[{_now()}] [seed {seed}] PCA fit done in {time.time() - t_pca:.1f}s", flush=True)

        t_tr = time.time()
        print(
            f"[{_now()}] [seed {seed}] PCA transform start: total_pixels={H*W} B={B0} -> {pca_bands}",
            flush=True,
        )
        cube = _apply_pca(cube, pca)
        print(
            f"[{_now()}] [seed {seed}] PCA transform done: cube={tuple(cube.shape)} "
            f"in {time.time() - t_tr:.1f}s",
            flush=True,
        )
    elif spec_method in ("none", "raw"):
        pca_bands = int(cube.shape[-1])
    else:
        raise ValueError(f"Unsupported spectral_method={spec_method}")

    # ---- normalization (train-only) ----
    norm_path = meta_dir / "norm_stats.npz"
    if norm_path.exists():
        npz = np.load(str(norm_path))
        mean = npz["mean"].astype(np.float32)
        std = npz["std"].astype(np.float32)
    else:
        t_norm = time.time()
        print(f"[{_now()}] [seed {seed}] compute_train_norm start", flush=True)
        mean, std = compute_train_norm(
            cube,
            train_idx,
            mean_global_blend=float(bcfg.get("norm_mean_global_blend", 0.0)),
            std_global_ratio=float(bcfg.get("norm_std_global_ratio", 0.05)),
            std_abs_floor=float(bcfg.get("norm_std_abs_floor", 1e-3)),
        )
        mean = mean.astype(np.float32, copy=False)
        std = std.astype(np.float32, copy=False)
        np.savez_compressed(norm_path, mean=mean, std=std)
        print(f"[{_now()}] [seed {seed}] compute_train_norm done in {time.time() - t_norm:.1f}s", flush=True)

    # ---- datasets / loaders ----
    batch_size = int(bcfg.get("batch_size", train_cfg.get("batch_size", 16)))
    eval_batch_size = int(bcfg.get("eval_batch_size", train_cfg.get("eval_batch_size", 256)))
    val_eval_batch_size = int(bcfg.get("val_eval_batch_size", eval_batch_size))
    test_eval_batch_size = int(bcfg.get("test_eval_batch_size", eval_batch_size))
    eval_infer_chunk_bs = int(bcfg.get("eval_infer_chunk_bs", 0))
    test_infer_chunk_bs = int(bcfg.get("test_infer_chunk_bs", eval_infer_chunk_bs))
    eval_max_batches = int(bcfg.get("eval_max_batches", 0))
    test_max_batches = int(bcfg.get("test_max_batches", 0))
    prefer_full_batch_eval = bool(bcfg.get("prefer_full_batch_eval", True))
    num_workers = int(bcfg.get("num_workers", train_cfg.get("num_workers", 0)))
    prefetch_factor = int(bcfg.get("prefetch_factor", 2))
    persistent_workers = bool(bcfg.get("persistent_workers", True))

    ds_train = HSIPatchDataset(
        cube=cube,
        gt=gt,
        indices=train_idx,
        patch_size=patch_size,
        mean=mean,
        std=std,
        label_offset=label_offset,
        augment=bool(bcfg.get("augment", False)),
        spec_dropout_p=float(bcfg.get("spectral_dropout", 0.0)),
        spec_dropout_ratio=float(bcfg.get("spectral_dropout_ratio", 0.0)),
        noise_std=float(bcfg.get("input_noise_std", 0.0)),
    )
    ds_val = HSIPatchDataset(
        cube=cube,
        gt=gt,
        indices=val_idx,
        patch_size=patch_size,
        mean=mean,
        std=std,
        label_offset=label_offset,
        augment=False,
    )
    ds_test = HSIPatchDataset(
        cube=cube,
        gt=gt,
        indices=test_idx,
        patch_size=patch_size,
        mean=mean,
        std=std,
        label_offset=label_offset,
        augment=False,
    )

    eval_num_workers = int(bcfg.get("eval_num_workers", max(2, min(num_workers, 4)) if num_workers > 0 else 0))
    test_num_workers = int(bcfg.get("test_num_workers", eval_num_workers))
    val_prefetch_factor = int(bcfg.get("val_prefetch_factor", 2))
    test_prefetch_factor = int(bcfg.get("test_prefetch_factor", 2))
    persistent_eval_workers = bool(bcfg.get("persistent_eval_workers", False))

    dl_train_kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dl_val_kwargs: Dict[str, Any] = dict(
        batch_size=val_eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dl_test_kwargs: Dict[str, Any] = dict(
        batch_size=test_eval_batch_size,
        shuffle=False,
        num_workers=test_num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if num_workers > 0:
        dl_train_kwargs["prefetch_factor"] = prefetch_factor
        dl_train_kwargs["persistent_workers"] = persistent_workers
    if eval_num_workers > 0:
        dl_val_kwargs["prefetch_factor"] = val_prefetch_factor
        dl_val_kwargs["persistent_workers"] = persistent_eval_workers
    if test_num_workers > 0:
        dl_test_kwargs["prefetch_factor"] = test_prefetch_factor
        dl_test_kwargs["persistent_workers"] = persistent_eval_workers

    dl_train = DataLoader(ds_train, **dl_train_kwargs)
    dl_val = DataLoader(ds_val, **dl_val_kwargs)
    dl_test = DataLoader(ds_test, **dl_test_kwargs)

    if len(dl_train) == 0:
        raise RuntimeError("Train DataLoader is empty. Check split_json train_indices and batch_size.")

    print(
        f"[{_now()}] [seed {seed}] dataloaders ready: "
        f"train_batches={len(dl_train)} val_batches={len(dl_val)} test_batches={len(dl_test)} "
        f"bs={batch_size} val_bs={val_eval_batch_size} test_bs={test_eval_batch_size} "
        f"workers={num_workers} eval_mode=one_batch_one_forward "
        f"eval_max_batches={eval_max_batches if eval_max_batches>0 else 'FULL'} "
        f"test_max_batches={test_max_batches if test_max_batches>0 else 'FULL'}",
        flush=True,
    )

    # ---- import external model ----
    ext_root = bcfg.get("external_path", None)
    ext_root = _resolve_path(ext_root) if ext_root else (repo_root / "external" / "baselines" / "3dss_mamba")
    if not ext_root.exists():
        raise FileNotFoundError(f"3dss_mamba external root not found: {ext_root}")

    sys.path.insert(0, str(ext_root))
    try:
        from models.videomamba import VisionMamba  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import VisionMamba from external 3dss_mamba. "
            f"Expected {ext_root}/models/videomamba.py to define VisionMamba. Error: {e}"
        ) from e

    # ---- model hyperparams ----
    embed_dim = int(bcfg.get("embed_dim", 32))
    depth = int(bcfg.get("depth", 1))
    d_state = int(bcfg.get("d_state", 16))
    dt_rank = int(bcfg.get("dt_rank", math.ceil(embed_dim / 16)))
    dim_inner = int(bcfg.get("dim_inner", 2 * embed_dim))
    group_type = str(bcfg.get("group_type", "Cube"))
    scan_type = str(bcfg.get("scan_type", "Parallel spectral-spatial"))
    k_group = int(bcfg.get("k_group", 4))
    pos = bool(bcfg.get("pos", False))
    cls = bool(bcfg.get("cls", False))

    conv3D_channel = int(bcfg.get("conv3D_channel", 32))
    ck = bcfg.get("conv3D_kernel", (3, 5, 5))
    if isinstance(ck, (list, tuple)) and len(ck) == 3:
        conv3D_kernel = tuple(int(x) for x in ck)
    else:
        raise ValueError("conv3D_kernel must be a list/tuple of 3 ints, e.g. [3,5,5]")

    dim_patch = int(bcfg.get("dim_patch", patch_size - conv3D_kernel[1] + 1))
    dim_linear = int(bcfg.get("dim_linear", pca_bands - conv3D_kernel[0] + 1))
    if dim_patch <= 0 or dim_linear <= 0:
        raise ValueError(
            f"Invalid dim_patch/dim_linear computed. "
            f"patch_size={patch_size}, pca_bands={pca_bands}, conv3D_kernel={conv3D_kernel} -> "
            f"dim_patch={dim_patch}, dim_linear={dim_linear}"
        )

    kwargs: Dict[str, Any] = dict(
        group_type=group_type,
        k_group=k_group,
        embed_dim=embed_dim,
        dt_rank=dt_rank,
        d_inner=dim_inner,
        d_state=d_state,
        num_classes=num_classes,
        depth=depth,
        scan_type=scan_type,
        pos=pos,
        cls=cls,
        conv3D_channel=conv3D_channel,
        conv3D_kernel=conv3D_kernel,
        dim_patch=dim_patch,
        dim_linear=dim_linear,
    )
    sig = inspect.signature(VisionMamba.__init__)
    filt = {k: v for k, v in kwargs.items() if k in sig.parameters}

    base_model = VisionMamba(**filt)
    model = _VMambaWrapper(base_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _enable_speed_flags(
        cudnn_benchmark=bool(bcfg.get("speed_cudnn_benchmark", bcfg.get("cudnn_benchmark", True))),
        tf32=bool(bcfg.get("speed_tf32", bcfg.get("tf32", True))),
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
        alloc = int(torch.cuda.memory_allocated() / (1024 * 1024))
        reserved = int(torch.cuda.memory_reserved() / (1024 * 1024))
        print(f"[{_now()}] [seed {seed}] cuda ready: allocated={alloc}MiB reserved={reserved}MiB", flush=True)

    lr = float(bcfg.get("lr", 3e-4))
    weight_decay = float(bcfg.get("weight_decay", 2e-2))
    betas = bcfg.get("betas", [0.9, 0.999])
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        betas_t = (float(betas[0]), float(betas[1]))
    else:
        betas_t = (0.9, 0.999)

    use_fused = bool(bcfg.get("speed_fused_adamw", bcfg.get("fused_adamw", True)))
    fused_used = False
    try:
        if use_fused and device.type == "cuda":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas_t, fused=True)
            fused_used = True
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas_t)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas_t)

    max_epochs = int(bcfg.get("max_epochs", 160))
    min_epochs = int(bcfg.get("min_epochs", 40))
    early_patience = int(bcfg.get("early_stop_patience", 20))
    select_metric = str(bcfg.get("select_metric", "kappa"))
    grad_clip = float(bcfg.get("grad_clip", 1.0))
    steps_per_epoch = int(bcfg.get("steps_per_epoch", 0))
    min_lr = float(bcfg.get("min_lr", 1e-6))
    log_every = int(bcfg.get("log_every", 10))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=min_lr,
    )

    label_smoothing = float(bcfg.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    use_amp = bool(args.use_amp and device.type == "cuda")
    scaler = (torch.amp.GradScaler('cuda', enabled=use_amp) if hasattr(torch, 'amp') else torch.cuda.amp.GradScaler(enabled=use_amp))

    print(
        f"[{_now()}] [seed {seed}] train start: epochs={max_epochs} min_epochs={min_epochs} "
        f"early_patience={early_patience} select_metric={select_metric} lr={lr} wd={weight_decay} "
        f"amp={use_amp} fused_adamw={fused_used} steps_per_epoch={'FULL' if steps_per_epoch<=0 else steps_per_epoch}",
        flush=True,
    )

    best_score = float("-inf")
    best_ep = -1
    bad = 0
    best_state: Optional[Dict[str, Any]] = None

    for ep in range(1, max_epochs + 1):
        ep_t0 = time.time()

        loss = train_one_epoch(
            model,
            dl_train,
            optimizer,
            device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=grad_clip,
            criterion=criterion,
            steps_per_epoch=steps_per_epoch,
        )

        val_out = _evaluate(
            model,
            dl_val,
            device,
            num_classes,
            use_amp=use_amp,
            infer_chunk_bs=eval_infer_chunk_bs,
            phase="val",
            log_every=0,
            seed=seed,
            max_batches=eval_max_batches,
            prefer_full_batch=prefer_full_batch_eval,
        )
        score = _select_score(val_out, select_metric)

        scheduler.step()

        improved = score > best_score + 1e-12
        if improved:
            best_score = score
            best_ep = ep
            bad = 0
            best_state = {"model": model.state_dict(), "ep": ep, "val": val_out, "score": score, "loss": loss}
        else:
            bad += 1

        if improved or ep == 1 or (log_every and ep % log_every == 0) or ep == max_epochs:
            lr_now = float(optimizer.param_groups[0]["lr"])
            t_ep = time.time() - ep_t0
            print(
                f"[{_now()}] [seed {seed}] [ep {ep:04d}] loss={loss:.6f} lr={lr_now:.3e} | "
                f"VAL OA={val_out['OA']:.4f} AA={val_out['AA']:.4f} Kappa={val_out['Kappa']:.4f} "
                f"score={score:.4f} bad={bad} time={t_ep:.1f}s",
                flush=True,
            )

        if ep >= min_epochs and bad >= early_patience:
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint captured.")

    model.load_state_dict(best_state["model"])
    val_best = best_state["val"]

    # Free train-only objects before the long dense test pass.
    try:
        del optimizer, scheduler, scaler, criterion, dl_train, ds_train
    except Exception:
        pass
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(
        f"[{_now()}] [seed {seed}] final test eval start: batches={len(dl_test)} "
        f"test_bs={test_eval_batch_size} subset_batches={test_max_batches if test_max_batches>0 else 'FULL'} "
        f"eval_mode=one_batch_one_forward",
        flush=True,
    )
    test_out = _evaluate(
        model,
        dl_test,
        device,
        num_classes,
        use_amp=use_amp,
        infer_chunk_bs=test_infer_chunk_bs,
        phase="test",
        log_every=int(bcfg.get("test_log_every", 50)),
        seed=seed,
        max_batches=test_max_batches,
        prefer_full_batch=prefer_full_batch_eval,
    )
    print(f"[{_now()}] [seed {seed}] final test eval done", flush=True)

    metrics = {
        "val": {"OA": float(val_best["OA"]), "AA": float(val_best["AA"]), "Kappa": float(val_best["Kappa"])},
        "test": {"OA": float(test_out["OA"]), "AA": float(test_out["AA"]), "Kappa": float(test_out["Kappa"])},
        "meta": {
            "dataset": dataset_name,
            "baseline": "3dss_mamba",
            "split_json": str(split_path),
            "seed": seed,
            "num_classes": num_classes,
            "label_offset": label_offset,
            "patch_size": patch_size,
            "spectral_method": spec_method,
            "pca_bands": int(pca_bands),
            "prefer_full_batch_eval": bool(prefer_full_batch_eval),
            "eval_max_batches": int(eval_max_batches),
            "test_max_batches": int(test_max_batches),
            "best_ep": int(best_ep),
            "best_val_score": float(best_score),
            "cfg_fingerprint": {
                "dataset_cfg_sha1": _sha1_file(dataset_cfg_path),
                "train_cfg_sha1": _sha1_file(train_cfg_path),
                "baseline_cfg_sha1": _sha1_file(baseline_cfg_path),
            },
            "time_sec": float(time.time() - t0),
        },
    }

    save_json(out_dir / "metrics.json", metrics)
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(ckpt_dir)
    torch.save({"model": model.state_dict(), "meta": metrics["meta"]}, ckpt_dir / "best.pt")

    print(f"[{_now()}] [seed {seed}] [done] out_dir={out_dir} best_ep={best_ep} val={metrics['val']} test={metrics['test']}", flush=True)


if __name__ == "__main__":
    main()
