#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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

sys.path.insert(0, str(repo_root))

from hsi3d.utils.io import load_yaml, load_json, ensure_dir, save_json  # noqa: E402
from hsi3d.utils.seed import set_global_seed  # noqa: E402
from hsi3d.data.hsi_dataset import HSIPatchDataset, compute_train_norm  # noqa: E402
from hsi3d.training.engine import train_one_epoch, evaluate  # noqa: E402


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()


def _get(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _ensure_hwb_cube(cube: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure cube is HxWxB and gt is HxW (matching).
    Accept common layouts:
      - cube: (H,W,B) or (B,H,W)
      - gt: (H,W) or (W,H)
    """
    if gt.ndim != 2:
        raise ValueError(f"gt must be 2D, got {gt.shape}")

    # fix gt orientation
    H, W = gt.shape
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got {cube.shape}")

    # case1: cube (H,W,B)
    if cube.shape[0] == H and cube.shape[1] == W:
        return cube, gt

    # case2: cube (W,H,B) -> transpose hw
    if cube.shape[0] == W and cube.shape[1] == H:
        return cube.transpose(1, 0, 2), gt

    # case3: cube (B,H,W)
    if cube.shape[1] == H and cube.shape[2] == W:
        return cube.transpose(1, 2, 0), gt

    # case4: cube (B,W,H)
    if cube.shape[1] == W and cube.shape[2] == H:
        return cube.transpose(2, 1, 0), gt

    raise ValueError(f"Unsupported cube/gt shapes: cube={cube.shape}, gt={gt.shape}")


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
            "scikit-learn is required for PCA in SSFTT baseline. "
            "Install it with: pip install scikit-learn"
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


def _load_external_ssfttnet(py_path: Path) -> Any:
    """
    Dynamically load SSFTTnet from a plain .py file.
    """
    if not py_path.exists():
        raise FileNotFoundError(f"SSFTTnet file not found: {py_path}")

    # Ensure local folder is importable for sibling imports inside SSFTTnet.py
    mod_dir = str(py_path.parent.resolve())
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)

    module = SourceFileLoader("ssftt_external", str(py_path)).load_module()  # type: ignore
    if not hasattr(module, "SSFTTnet"):
        raise AttributeError(f"{py_path} does not define SSFTTnet")
    return module.SSFTTnet


class _SSFTTWrapper(nn.Module):
    """
    Make SSFTTnet compatible with our engine:
      - accept (x) or (x, x_spec)
      - x is (B, Bands, P, P) -> SSFTT expects (B, 1, Bands, P, P)
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor, x_spec: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x as (B,Bands,P,P), got {tuple(x.shape)}")
        x5 = x.unsqueeze(1)  # (B,1,Bands,P,P)
        return self.base(x5)


def _select_score(d: Dict[str, Any], metric: str) -> float:
    m = str(metric).lower().strip()
    if m in ("kappa", "k"):
        return float(d.get("Kappa", 0.0))
    if m in ("oa",):
        return float(d.get("OA", 0.0))
    if m in ("aa",):
        return float(d.get("AA", 0.0))
    if m in ("loss", "val_loss"):
        # engine.evaluate does not return loss; treat as -inf
        return float("-inf")
    # default
    return float(d.get("Kappa", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", type=str, required=True)
    ap.add_argument("--train_cfg", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--baseline", type=str, default="ssftt")
    ap.add_argument("--patch_size", type=int, default=15)
    ap.add_argument("--use_amp", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    meta_dir = out_dir / "meta"
    ensure_dir(meta_dir)

    # ---- seed (compatible with your repo set_global_seed signature) ----
    seed = int(args.seed)
    # Keep seed control but avoid hard deterministic constraints that can crash on Conv3d/cuBLAS.
    try:
        set_global_seed(seed, deterministic=True)  # type: ignore[arg-type]
    except TypeError:
        set_global_seed(seed)

    # Scheme-A: practical reproducibility (stable seeds) without strict deterministic algorithms.
    # Strict mode may raise on some CUDA ops (e.g., conv3d paths using cuBLAS).
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    dataset_cfg = load_yaml(_resolve_path(args.dataset_cfg))
    train_cfg = load_yaml(_resolve_path(args.train_cfg))
    split = load_json(_resolve_path(args.split_json))

    dataset_name = str(dataset_cfg.get("dataset", dataset_cfg.get("name", "dataset"))).strip()
    label_offset = int(dataset_cfg.get("label_offset", 1))
    num_classes = int(dataset_cfg.get("num_classes", dataset_cfg.get("n_classes", 0)))
    if num_classes <= 0:
        raise ValueError(f"num_classes missing in dataset cfg: {args.dataset_cfg}")

    # locate baseline cfg file
    # pavia_university uses pu.yaml in this repo layout
    base_tag = "pu" if dataset_name in ("pavia_university", "paviau", "pu") else dataset_name
    baseline_cfg_path = repo_root / "configs" / "baselines" / f"{base_tag}.yaml"
    baseline_all = load_yaml(baseline_cfg_path)
    ssftt_cfg = _get(baseline_all, ("baseline", "ssftt"), None)
    if ssftt_cfg is None and isinstance(baseline_all, dict) and "ssftt" in baseline_all:
        ssftt_cfg = baseline_all["ssftt"]
    if ssftt_cfg is None:
        raise ValueError(f"Cannot find baseline.ssftt in {baseline_cfg_path}")

    patch_size = int(args.patch_size)
    if patch_size != int(ssftt_cfg.get("patch_size", patch_size)):
        raise ValueError(f"patch_size mismatch: cli={patch_size}, cfg={ssftt_cfg.get('patch_size')}")

    spec_method = str(ssftt_cfg.get("spectral_method", "pca")).lower().strip()
    pca_bands = int(ssftt_cfg.get("pca_bands", 30))
    if spec_method == "pca" and pca_bands != 30:
        # external SSFTTnet is commonly hard-coded to 30 -> 28 after conv3d
        raise ValueError(
            f"External SSFTTnet typically requires pca_bands=30, got {pca_bands}. "
            "If you want other bands, you must modify the external SSFTTnet conv2d in_channels."
        )

    batch_size = int(ssftt_cfg.get("batch_size", train_cfg.get("batch_size", 16)))
    eval_batch_size = int(ssftt_cfg.get("eval_batch_size", train_cfg.get("eval_batch_size", 256)))
    num_workers = int(ssftt_cfg.get("num_workers", train_cfg.get("num_workers", 0)))

    lr = float(ssftt_cfg.get("lr", 3e-4))
    weight_decay = float(ssftt_cfg.get("weight_decay", 1e-2))
    max_epochs = int(ssftt_cfg.get("max_epochs", 200))
    min_epochs = int(ssftt_cfg.get("min_epochs", 20))
    early_patience = int(ssftt_cfg.get("early_stop_patience", 30))
    select_metric = str(ssftt_cfg.get("select_metric", "kappa"))

    # model hypers (passed only if supported)
    num_tokens = int(ssftt_cfg.get("num_tokens", 4))
    dim = int(ssftt_cfg.get("dim", 64))
    depth = int(ssftt_cfg.get("depth", 1))
    heads = int(ssftt_cfg.get("heads", 8))
    mlp_dim = int(ssftt_cfg.get("mlp_dim", 128))
    dropout = float(ssftt_cfg.get("dropout", 0.1))
    emb_dropout = float(ssftt_cfg.get("emb_dropout", 0.1))

    # ---- load cube/gt ----
    processed_dir = _resolve_path(str(dataset_cfg.get("processed_dir", "")))
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")
    cube_path = processed_dir / "cube.npy"
    gt_path = processed_dir / "gt.npy"
    cube = np.load(cube_path)
    gt = np.load(gt_path)
    cube, gt = _ensure_hwb_cube(cube, gt)

    H, W, B = cube.shape

    train_idx = np.asarray(split.get("train_indices", []))
    val_idx = np.asarray(split.get("val_indices", []))
    test_idx = np.asarray(split.get("test_indices", []))
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"split_json missing indices: {args.split_json}")

    train_flat = _flat_indices(train_idx, H, W)
    # ---- spectral method ----
    if spec_method == "pca":
        whiten = bool(ssftt_cfg.get("pca_whiten", False))
        pca = _fit_pca_train_only(cube, train_flat, n_components=pca_bands, whiten=whiten, random_state=seed)
        cube = _apply_pca(cube, pca)  # (H,W,pca_bands)
    elif spec_method in ("none", "raw"):
        pass
    else:
        raise ValueError(f"Unsupported spectral_method={spec_method}")


    # ---- normalization stats (train-only) on *current cube* (after spectral_method) ----
    norm_path = meta_dir / "norm_stats.npz"
    if norm_path.exists():
        npz = np.load(str(norm_path))
        mean = npz["mean"].astype(np.float32)
        std = npz["std"].astype(np.float32)
    else:
        mean, std = compute_train_norm(
            cube,
            train_idx,
            mean_global_blend=float(ssftt_cfg.get("norm_mean_global_blend", 0.0)),
            std_global_ratio=float(ssftt_cfg.get("norm_std_global_ratio", 0.05)),
            std_abs_floor=float(ssftt_cfg.get("norm_std_abs_floor", 1e-3)),
        )
        mean = mean.astype(np.float32, copy=False)
        std = std.astype(np.float32, copy=False)
        np.savez_compressed(norm_path, mean=mean, std=std)


    # ---- datasets / loaders ----
    ds_train = HSIPatchDataset(
        cube=cube,
        gt=gt,
        indices=train_idx,
        patch_size=patch_size,
        mean=mean,
        std=std,
        label_offset=label_offset,
        augment=bool(ssftt_cfg.get("augment", False)),
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

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if len(dl_train) == 0:
        raise RuntimeError("Train DataLoader is empty. Check split_json train_indices and batch_size.")

    # ---- model load ----
    ext_path = ssftt_cfg.get("external_path", None)
    if ext_path:
        ssftt_py = _resolve_path(str(ext_path))
    else:
        ssftt_py = repo_root / "external" / "baselines" / "ssftt" / "cls_SSFTT_IP" / "SSFTTnet.py"

    SSFTTnet = _load_external_ssfttnet(ssftt_py)

    # Instantiate external model (pass only supported args)
    kwargs: Dict[str, Any] = dict(
        in_channels=1,
        num_classes=num_classes,
        num_tokens=num_tokens,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout,
    )
    # external signature may differ; filter unknown kwargs
    import inspect

    sig = inspect.signature(SSFTTnet.__init__)
    filt = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            filt[k] = v

    base_model = SSFTTnet(**filt)
    model = _SSFTTWrapper(base_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # simple cosine anneal
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=float(ssftt_cfg.get("min_lr", 1e-6)))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp and device.type == "cuda"))

    best_score = float("-inf")
    best_ep = -1
    bad = 0
    best_state: Optional[Dict[str, Any]] = None

    for ep in range(1, max_epochs + 1):
        _ = train_one_epoch(
            model,
            dl_train,
            optimizer,
            device,
            scaler=scaler,
            use_amp=bool(args.use_amp and device.type == "cuda"),
            grad_clip=float(ssftt_cfg.get("grad_clip", 0.0)),
            label_smoothing=float(ssftt_cfg.get("label_smoothing", 0.0)),
            spec_dropout_p=float(ssftt_cfg.get("spectral_dropout", 0.0)),
            spec_dropout_ratio=float(ssftt_cfg.get("spectral_dropout_ratio", 0.0)),
            aug_noise_std=float(ssftt_cfg.get("input_noise_std", 0.0)),
            patch_size=patch_size,
            steps_per_epoch=int(ssftt_cfg.get("steps_per_epoch", 0)),
            epoch=ep,
        )

        val_out = evaluate(
            model,
            dl_val,
            device,
            num_classes=num_classes,
            use_amp=bool(args.use_amp and device.type == "cuda"),
            patch_size=patch_size,
            steps=int(ssftt_cfg.get("val_steps", 0)),
        )
        score = _select_score(val_out, select_metric)

        scheduler.step()

        improved = score > best_score + 1e-12
        if improved:
            best_score = score
            best_ep = ep
            bad = 0
            best_state = {"model": model.state_dict(), "ep": ep, "val": val_out, "score": score}
        else:
            bad += 1

        if ep >= min_epochs and bad >= early_patience:
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint captured.")

    # restore best
    model.load_state_dict(best_state["model"])

    val_best = best_state["val"]
    test_out = evaluate(
        model,
        dl_test,
        device,
        num_classes=num_classes,
        use_amp=bool(args.use_amp and device.type == "cuda"),
        patch_size=patch_size,
        steps=int(ssftt_cfg.get("test_steps", 0)),
    )

    metrics = {
        "val": {"OA": float(val_best["OA"]), "AA": float(val_best["AA"]), "Kappa": float(val_best["Kappa"])},
        "test": {"OA": float(test_out["OA"]), "AA": float(test_out["AA"]), "Kappa": float(test_out["Kappa"])},
        "meta": {
            "dataset": dataset_name,
            "baseline": "ssftt",
            "split_json": str(_resolve_path(args.split_json)),
            "seed": seed,
            "num_classes": num_classes,
            "label_offset": label_offset,
            "patch_size": patch_size,
            "spectral_method": spec_method,
            "pca_bands": pca_bands if spec_method == "pca" else None,
            "best_ep": int(best_ep),
            "best_val_score": float(best_score),
            "time_sec": float(time.time() - t0),
        },
    }

    save_json(out_dir / "metrics.json", metrics)
    # optional: save best checkpoint for inspection
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(ckpt_dir)
    torch.save({"model": model.state_dict(), "meta": metrics["meta"]}, ckpt_dir / "best.pt")

    print(f"[done] out_dir={out_dir} best_ep={best_ep} val={metrics['val']} test={metrics['test']}")



if __name__ == "__main__":
    main()
