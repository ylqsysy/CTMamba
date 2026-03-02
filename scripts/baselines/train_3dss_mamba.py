#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "configs").exists() and (p / "src" / "hsi3d").exists():
            return p
        p = p.parent
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
    if cube.shape[0] == H and cube.shape[1] == W:
        return cube, gt
    if cube.shape[1] == H and cube.shape[2] == W:
        return np.transpose(cube, (1, 2, 0)), gt
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


def _fit_pca_train_only(cube_hwb: np.ndarray, train_flat_idx: np.ndarray, n_components: int, *, whiten: bool, random_state: int) -> Any:
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


def classification_metrics_from_preds(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> Dict[str, float]:
    y_true_np = torch.as_tensor(y_true).detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    y_pred_np = torch.as_tensor(y_pred).detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    _cm_update(cm, y_true_np, y_pred_np, int(num_classes))
    oa, aa, kappa = _cm_scores(cm)
    return {"OA": float(oa), "AA": float(aa), "Kappa": float(kappa)}


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
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_batches_total = len(dl)
    n_batches = n_batches_total if int(max_batches) <= 0 else min(n_batches_total, int(max_batches))
    t_eval0 = time.time()

    def _forward_chunks(xb_all: torch.Tensor, chunk_bs: int) -> torch.Tensor:
        if chunk_bs <= 0 or chunk_bs >= int(xb_all.shape[0]):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                return model(xb_all)
        outs = []
        for s in range(0, int(xb_all.shape[0]), int(chunk_bs)):
            xb = xb_all[s : s + int(chunk_bs)]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outs.append(model(xb))
        return torch.cat(outs, dim=0)

    for bi, (x, _x_spec, y) in enumerate(dl, start=1):
        if bi > n_batches:
            break
        if x.device.type != device.type:
            x = x.to(device, non_blocking=True)
        x = x.unsqueeze(1)

        if int(infer_chunk_bs) > 0:
            logits = _forward_chunks(x, int(infer_chunk_bs))
        else:
            try:
                if prefer_full_batch:
                    logits = _forward_chunks(x, 0)
                else:
                    logits = _forward_chunks(x, min(64, int(x.shape[0])))
            except torch.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                fallback = min(64, int(x.shape[0]))
                ok = False
                while fallback >= 1:
                    try:
                        logits = _forward_chunks(x, fallback)
                        ok = True
                        break
                    except torch.OutOfMemoryError:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        fallback //= 2
                if not ok:
                    raise

        pred = logits.argmax(dim=1).detach().cpu().numpy()
        yt = y.detach().cpu().numpy()
        _cm_update(cm, yt, pred, num_classes)

        if log_every and (bi % int(log_every) == 0 or bi == n_batches):
            took = time.time() - t_eval0
            spb = took / max(1, bi)
            eta = spb * (n_batches - bi)
            seed_txt = f"[seed {seed}] " if seed is not None else ""
            print(f"[{_now()}] {seed_txt}[{phase}] batch {bi}/{n_batches} elapsed={took:.1f}s eta={eta:.1f}s", flush=True)

    oa, aa, kappa = _cm_scores(cm)
    return {"OA": oa, "AA": aa, "Kappa": kappa}


def _build_padded_cube(cube: np.ndarray, patch_size: int, pad_mode: str) -> np.ndarray:
    pad = patch_size // 2
    mode = str(pad_mode or "reflect").lower()
    if mode not in {"reflect", "edge", "constant", "symmetric"}:
        mode = "reflect"
    if pad <= 0:
        return cube
    if mode == "constant":
        return np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode=mode, constant_values=0)
    return np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode=mode)


@torch.no_grad()
def _evaluate_fast_indices(
    model: nn.Module, cube_pad: np.ndarray, gt: np.ndarray, flat_indices: np.ndarray,
    label_offset: int, patch_size: int, device: torch.device, num_classes: int,
    use_amp: bool, phase: str, seed: int, log_every: int = 0, chunk_bs: int = 4096,
) -> Dict[str, float]:
    model.eval()
    if flat_indices.size == 0:
        return {"OA": 0.0, "AA": 0.0, "Kappa": 0.0}

    H, W = gt.shape
    gt_flat = gt.reshape(-1)
    y_true_all = gt_flat[flat_indices].astype(np.int64) - int(label_offset)

    cube_t = torch.from_numpy(cube_pad).to(device)
    flat_indices_t = torch.from_numpy(flat_indices).to(device)
    
    total = int(flat_indices.size)
    ptr = 0
    preds: list[np.ndarray] = []
    t0 = time.time()
    cur_bs = max(1, int(chunk_bs))
    log_counter = 0

    pad = patch_size // 2
    W_orig = cube_pad.shape[1] - 2 * pad
    off = torch.arange(-pad, pad + 1, device=device)

    while ptr < total:
        end = min(ptr + cur_bs, total)
        idx_chunk = flat_indices_t[ptr:end]
        
        r = torch.div(idx_chunk, W_orig, rounding_mode='floor') + pad
        c = idx_chunk % W_orig + pad
        rr = r[:, None, None] + off[None, :, None]
        cc = c[:, None, None] + off[None, None, :]
        
        x = cube_t[rr, cc].permute(0, 3, 1, 2).unsqueeze(1)
        
        try:
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(x)
            else:
                logits = model(x)
        except RuntimeError as e:
            msg = str(e).lower()
            if device.type == 'cuda' and ('out of memory' in msg or 'cuda' in msg) and cur_bs > 16:
                torch.cuda.empty_cache()
                cur_bs = max(16, cur_bs // 2)
                continue
            raise
            
        preds.append(logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
        ptr = end
        done = ptr
        log_counter += 1

        if log_every and (log_counter % log_every == 0 or done >= total):
            elapsed = time.time() - t0
            rate = done / max(1e-6, elapsed)
            eta = (total - done) / max(1e-6, rate)
            print(f"[{_now()}] [seed {seed}] [{phase}] pix {done}/{total} chunk_bs={cur_bs} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    y_pred_all = np.concatenate(preds, axis=0)
    return classification_metrics_from_preds(torch.from_numpy(y_true_all), torch.from_numpy(y_pred_all), num_classes=num_classes)


def _select_score(m: Dict[str, float], select_metric: str) -> float:
    sel = str(select_metric).lower().strip()
    if sel in ("kappa", "kap"): return float(m["Kappa"])
    if sel in ("oa",): return float(m["OA"])
    if sel in ("aa",): return float(m["AA"])
    if sel in ("min", "min3", "min_oa_aa_kappa"): return float(min(m["OA"], m["AA"], m["Kappa"]))
    return float(m["Kappa"])


class _VMambaWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        return out[0] if isinstance(out, (tuple, list)) else out


class _TensorTripletDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y
        self.x_spec = torch.zeros((x.shape[0], 1), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.x_spec[idx], self.y[idx]


def _cache_dataset_to_memory(ds: Dataset, seed: int, tag: str) -> Dataset:
    t0 = time.time()
    xs, ys = [], []
    n = len(ds)
    for i in range(n):
        x, _xspec, y = ds[i]
        xt = torch.from_numpy(x) if isinstance(x, np.ndarray) else (x.detach().cpu() if torch.is_tensor(x) else torch.tensor(x))
        y = y if torch.is_tensor(y) else torch.tensor(y)
        xs.append(xt.to(torch.float32).contiguous())
        ys.append(y.to(torch.long).view(()))
    
    x_all = torch.stack(xs, dim=0)
    y_all = torch.stack(ys, dim=0)
    
    print(f"[{_now()}] [seed {seed}] cached {tag} patches to CPU RAM in {time.time() - t0:.2f}s", flush=True)
    return _TensorTripletDataset(x_all, y_all)


def _enable_speed_flags(*, cudnn_benchmark: bool, tf32: bool) -> None:
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if tf32 and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def train_one_epoch(
    model: nn.Module, dl: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,
    *, scaler: torch.cuda.amp.GradScaler, use_amp: bool, grad_clip: float, criterion: nn.Module, steps_per_epoch: int,
    apply_gpu_aug: bool = True
) -> float:
    model.train()
    total_loss, n_seen = 0.0, 0

    for step, (x, _x_spec, y) in enumerate(dl, start=1):
        x = x.to(device, non_blocking=True).unsqueeze(1)
        y = y.to(device, non_blocking=True)

        if apply_gpu_aug:
            if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[3])
            if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[4])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0: x = torch.rot90(x, k, dims=[3, 4])

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
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")
    
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

    base_tag = _cfg_name(dataset_name)
    baseline_cfg_path = repo_root / "configs" / "baselines" / f"{base_tag}.yaml"
    baseline_all = load_yaml(baseline_cfg_path)
    bcfg = _get(baseline_all, ("baseline", "3dss_mamba"), None) or baseline_all.get("3dss_mamba", {})

    patch_size = int(args.patch_size)

    processed_dir = _resolve_path(str(dataset_cfg.get("processed_dir", "")))
    cube, gt = _ensure_hwb_cube(np.load(processed_dir / "cube.npy"), np.load(processed_dir / "gt.npy"))
    H, W, B0 = cube.shape

    train_idx = np.asarray(split.get("train_indices", []))
    val_idx = np.asarray(split.get("val_indices", []))
    test_idx = np.asarray(split.get("test_indices", []))
    train_flat = _flat_indices(train_idx, H, W)

    spec_method = str(bcfg.get("spectral_method", "pca")).lower().strip()
    pca_bands = int(bcfg.get("pca_bands", 15))
    if spec_method == "pca":
        pca = _fit_pca_train_only(cube, train_flat, n_components=pca_bands, whiten=bool(bcfg.get("pca_whiten", False)), random_state=seed)
        cube = _apply_pca(cube, pca)
    else:
        pca_bands = int(cube.shape[-1])

    norm_path = meta_dir / "norm_stats.npz"
    if norm_path.exists():
        npz = np.load(str(norm_path))
        mean, std = npz["mean"].astype(np.float32), npz["std"].astype(np.float32)
    else:
        mean, std = compute_train_norm(
            cube, train_idx, mean_global_blend=float(bcfg.get("norm_mean_global_blend", 0.0)),
            std_global_ratio=float(bcfg.get("norm_std_global_ratio", 0.05)), std_abs_floor=float(bcfg.get("norm_std_abs_floor", 1e-3))
        )
        mean, std = mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)
        np.savez_compressed(norm_path, mean=mean, std=std)

    batch_size = int(bcfg.get("batch_size", train_cfg.get("batch_size", 16)))
    val_eval_batch_size = int(bcfg.get("val_eval_batch_size", 256))
    test_eval_batch_size = int(bcfg.get("test_eval_batch_size", 256))
    num_workers = int(bcfg.get("num_workers", train_cfg.get("num_workers", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds_train = HSIPatchDataset(
        cube=cube, gt=gt, indices=train_idx, patch_size=patch_size, mean=mean, std=std,
        label_offset=label_offset, augment=bool(bcfg.get("augment", False)),
        spec_dropout_p=float(bcfg.get("spectral_dropout", 0.0)),
    )
    ds_val = HSIPatchDataset(cube=cube, gt=gt, indices=val_idx, patch_size=patch_size, mean=mean, std=std, label_offset=label_offset, augment=False)
    ds_test = HSIPatchDataset(cube=cube, gt=gt, indices=test_idx, patch_size=patch_size, mean=mean, std=std, label_offset=label_offset, augment=False)

    cube_pad_eval, val_idx_np, test_idx_np = None, None, None
    fast_eval_vectorized = bool(bcfg.get("fast_eval_vectorized", True))
    if fast_eval_vectorized:
        cube_eval = (cube.astype(np.float32, copy=False) - mean.reshape(1, 1, -1)) / np.maximum(std.reshape(1, 1, -1), 1e-6)
        cube_pad_eval = _build_padded_cube(cube_eval.astype(np.float32, copy=False), patch_size=patch_size, pad_mode=str(bcfg.get("pad_mode", "reflect")))
        val_idx_np, test_idx_np = np.asarray(val_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)

    if bool(bcfg.get("cache_train_patches", True)) and not bool(bcfg.get("augment", False)):
        ds_train = _cache_dataset_to_memory(ds_train, seed=seed, tag="train")

    dl_train_kwargs = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=bool(bcfg.get("pin_memory_train", False)), drop_last=False)
    if num_workers > 0:
        dl_train_kwargs.update({"prefetch_factor": int(bcfg.get("prefetch_factor", 2)), "persistent_workers": bool(bcfg.get("persistent_workers", True))})

    dl_val_kwargs = dict(batch_size=val_eval_batch_size, shuffle=False, num_workers=int(bcfg.get("eval_num_workers", 0)), pin_memory=bool(bcfg.get("pin_memory_eval", False)))
    dl_test_kwargs = dict(batch_size=test_eval_batch_size, shuffle=False, num_workers=int(bcfg.get("test_num_workers", 0)), pin_memory=bool(bcfg.get("pin_memory_eval", False)))

    dl_train = DataLoader(ds_train, **dl_train_kwargs)
    dl_val = DataLoader(ds_val, **dl_val_kwargs)
    dl_test = DataLoader(ds_test, **dl_test_kwargs)

    ext_root = _resolve_path(bcfg.get("external_path", repo_root / "external" / "baselines" / "3dss_mamba"))
    sys.path.insert(0, str(ext_root))
    from models.videomamba import VisionMamba  # type: ignore

    conv3D_kernel = tuple(int(x) for x in bcfg.get("conv3D_kernel", (3, 5, 5)))
    dim_patch = int(bcfg.get("dim_patch", patch_size - conv3D_kernel[1] + 1))
    dim_linear = int(bcfg.get("dim_linear", pca_bands - conv3D_kernel[0] + 1))

    kwargs = dict(
        group_type=str(bcfg.get("group_type", "Cube")), k_group=int(bcfg.get("k_group", 4)),
        embed_dim=int(bcfg.get("embed_dim", 32)), dt_rank=int(bcfg.get("dt_rank", 4)),
        d_inner=int(bcfg.get("dim_inner", 64)), d_state=int(bcfg.get("d_state", 16)),
        num_classes=num_classes, depth=int(bcfg.get("depth", 2)),
        scan_type=str(bcfg.get("scan_type", "Parallel spectral-spatial")),
        pos=bool(bcfg.get("pos", False)), cls=bool(bcfg.get("cls", False)),
        conv3D_channel=int(bcfg.get("conv3D_channel", 32)), conv3D_kernel=conv3D_kernel,
        dim_patch=dim_patch, dim_linear=dim_linear,
    )
    sig = inspect.signature(VisionMamba.__init__)
    base_model = VisionMamba(**{k: v for k, v in kwargs.items() if k in sig.parameters})
    
    model = _VMambaWrapper(base_model).to(device)

    _enable_speed_flags(cudnn_benchmark=bool(bcfg.get("speed_cudnn_benchmark", True)), tf32=bool(bcfg.get("speed_tf32", True)))

    lr, weight_decay = float(bcfg.get("lr", 3e-4)), float(bcfg.get("weight_decay", 2e-2))
    betas_t = tuple(float(x) for x in bcfg.get("betas", [0.9, 0.999]))
    
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas_t, fused=bool(bcfg.get("speed_fused_adamw", True)))
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas_t)

    max_epochs = int(bcfg.get("max_epochs", 150))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=float(bcfg.get("min_lr", 1e-6)))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(bcfg.get("label_smoothing", 0.0)))
    use_amp = bool(args.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if hasattr(torch, 'amp') else torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score, best_ep, bad, best_state = float("-inf"), -1, 0, None
    eval_every = max(1, int(bcfg.get("eval_every", 1)))
    steps_per_epoch = int(bcfg.get("steps_per_epoch", 0))

    for ep in range(1, max_epochs + 1):
        loss = train_one_epoch(model, dl_train, optimizer, device, scaler=scaler, use_amp=use_amp, grad_clip=float(bcfg.get("grad_clip", 1.0)), criterion=criterion, steps_per_epoch=steps_per_epoch, apply_gpu_aug=True)
        
        do_eval = (eval_every <= 1) or (ep == max_epochs) or (bool(bcfg.get("eval_on_first", True)) and ep == 1) or (ep % eval_every == 0)
        improved = False

        if do_eval:
            if fast_eval_vectorized and cube_pad_eval is not None and val_idx_np is not None:
                val_out = _evaluate_fast_indices(model, cube_pad=cube_pad_eval, gt=gt, flat_indices=val_idx_np, label_offset=label_offset, patch_size=patch_size, device=device, num_classes=num_classes, use_amp=use_amp, phase="val", seed=seed, log_every=0, chunk_bs=max(256, int(bcfg.get("fast_eval_chunk_bs", 512))))
            else:
                val_out = _evaluate(model, dl_val, device, num_classes, use_amp=use_amp, phase="val", seed=seed)
            
            score = _select_score(val_out, str(bcfg.get("select_metric", "kappa")))
            improved = score > best_score + 1e-12
            
            if improved:
                best_score, best_ep, bad, best_state = score, ep, 0, {"model": model.state_dict(), "ep": ep, "val": val_out, "score": score}
            else:
                bad += 1

        scheduler.step()
        
        if do_eval and (improved or ep == 1 or ep % int(bcfg.get("log_every", 10)) == 0):
            print(f"[{_now()}] [seed {seed}] [ep {ep:04d}] loss={loss:.6f} | VAL OA={val_out['OA']:.4f} AA={val_out['AA']:.4f} Kappa={val_out['Kappa']:.4f} score={score:.4f}", flush=True)
            if ep >= int(bcfg.get("min_epochs", 40)) and bad >= int(bcfg.get("early_stop_patience", 20)):
                break

    model.load_state_dict(best_state["model"])
    
    print(f"[{_now()}] [seed {seed}] Starting final test evaluation...", flush=True)
    if test_idx_np is not None:
        test_out = _evaluate_fast_indices(model, cube_pad=cube_pad_eval, gt=gt, flat_indices=test_idx_np, label_offset=label_offset, patch_size=patch_size, device=device, num_classes=num_classes, use_amp=use_amp, phase="test", seed=seed, log_every=int(bcfg.get("test_log_every", 20)), chunk_bs=max(256, int(bcfg.get("fast_eval_chunk_bs", 512))))
    else:
        test_out = _evaluate(model, dl_test, device, num_classes, use_amp=use_amp, phase="test", seed=seed, log_every=int(bcfg.get("test_log_every", 20)))

    metrics = {
        "val": {"OA": float(best_state["val"]["OA"]), "AA": float(best_state["val"]["AA"]), "Kappa": float(best_state["val"]["Kappa"])},
        "test": {"OA": float(test_out["OA"]), "AA": float(test_out["AA"]), "Kappa": float(test_out["Kappa"])},
        "meta": {"dataset": dataset_name, "best_ep": int(best_ep), "time_sec": float(time.time() - t0)},
    }
    
    save_json(out_dir / "metrics.json", metrics)
    ensure_dir(out_dir / "checkpoints")
    torch.save({"model": model.state_dict(), "meta": metrics["meta"]}, out_dir / "checkpoints" / "best.pt")
    print(f"[{_now()}] [seed {seed}] [done] test OA={metrics['test']['OA']:.4f} AA={metrics['test']['AA']:.4f} Kappa={metrics['test']['Kappa']:.4f}", flush=True)


if __name__ == "__main__":
    main()