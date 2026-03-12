#!/usr/bin/env python3
"""Evaluate a checkpoint on the validation and test splits."""

from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path
from typing import Dict, Any
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from utils.io import load_yaml, load_json, ensure_dir, save_json
from utils.seed import set_global_seed
from utils.hsi_dataset import HSIPatchDataset, compute_train_norm
from utils.hsi_preprocess import fit_and_apply_spectral_preprocess, load_spectral_preprocess, apply_spectral_preprocess
from utils.engine import evaluate as engine_evaluate
from models import CTMambaConfig, CTMambaModel


def _sha1_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _filter_kwargs_by_signature(fn_or_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn_or_cls)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _torch_load_compat(path: str | Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _make_model(model_cfg: Dict[str, Any], num_classes: int, raw_bands: int) -> torch.nn.Module:
    cfg_in = dict(model_cfg or {})
    alias = {
        "in_chans": "raw_bands",
        "in_channels": "raw_bands",
        "bands": "raw_bands",
        "n_bands": "raw_bands",
        "classes": "num_classes",
        "n_classes": "num_classes",
    }
    for k in list(cfg_in.keys()):
        if k in alias:
            cfg_in[alias[k]] = cfg_in.pop(k)

    cfg_in["num_classes"] = int(num_classes)
    cfg_in["raw_bands"] = int(raw_bands)

    allowed = set(getattr(CTMambaConfig, "__annotations__", {}).keys())
    unknown = sorted([k for k in cfg_in.keys() if k not in allowed])
    if unknown:
        raise ValueError(
            f"Unknown model config keys: {unknown}. "
            f"Allowed keys: {sorted(allowed)}"
        )
    cfg = CTMambaConfig(**cfg_in)
    return CTMambaModel(cfg)


def _resolve_num_classes(split: Dict[str, Any], dataset_cfg: Dict[str, Any], gt: np.ndarray, label_offset: int) -> int:
    fallback = int(gt.max()) - int(label_offset) + 1
    num_classes = int(split.get("num_classes", dataset_cfg.get("num_classes", fallback)))
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")
    return int(num_classes)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_cfg", required=True)
    p.add_argument("--model_cfg", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--data_root", default="data")
    p.add_argument("--split_tag", default="random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt_key", default="model")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--log_interval", type=int, default=0, help="Print eval progress every N steps (0=off).")
    p.add_argument("--train_cfg", default="", help="Optional train config path (for hash metadata consistency checks).")
    p.add_argument("--out", required=True)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    t0 = time.time()
    set_global_seed(int(args.seed))

    dcfg = load_yaml(args.dataset_cfg)
    mcfg = load_yaml(args.model_cfg)
    split = load_json(args.split_json)
    train_cfg = load_yaml(args.train_cfg) if str(args.train_cfg).strip() else {}

    dataset = str(dcfg.get("dataset", split.get("dataset", ""))).strip()
    if not dataset:
        raise SystemExit("[ERROR] dataset name not found in dataset_cfg/split_json.")

    raw_dir = Path(args.data_root) / "processed" / dataset / "raw"
    cube = np.load(raw_dir / "cube.npy")
    gt = np.load(raw_dir / "gt.npy")

    label_offset = int(split.get("label_offset", dcfg.get("label_offset", 1)))
    num_classes = _resolve_num_classes(split, dcfg, gt, label_offset)

    tr_idx = np.asarray(split.get("train_indices", split.get("train", [])), dtype=np.int64)
    va_idx = np.asarray(split.get("val_indices", split.get("val", [])), dtype=np.int64)
    te_idx = np.asarray(split.get("test_indices", split.get("test", [])), dtype=np.int64)

    patch_size = int(mcfg.get("patch_size", dcfg.get("patch_size", 15)))
    ckpt_path = Path(args.ckpt)
    out_dir = ckpt_path.parent.parent
    spectral_preprocess_path = out_dir / "meta" / "spectral_preprocess.npz"
    spectral_state: Dict[str, Any]
    if spectral_preprocess_path.exists():
        spectral_state = load_spectral_preprocess(spectral_preprocess_path)
        cube = apply_spectral_preprocess(cube, spectral_state)
    else:
        cube, spectral_state = fit_and_apply_spectral_preprocess(
            cube,
            tr_idx,
            train_cfg,
            gt_shape=gt.shape,
            save_path=None,
        )

    norm_path = out_dir / "meta" / "norm_stats.npz"
    if norm_path.exists():
        z = np.load(norm_path)
        mean = z["mean"].astype(np.float32)
        std = z["std"].astype(np.float32)
    else:
        mean, std = compute_train_norm(
            cube,
            tr_idx,
            mean_global_blend=float(train_cfg.get("norm_mean_global_blend", 0.0)),
            std_global_ratio=float(train_cfg.get("norm_std_global_ratio", 0.0)),
            std_abs_floor=float(train_cfg.get("norm_std_abs_floor", 1.0e-3)),
        )

    common = {
        "cube": cube,
        "gt": gt,
        "patch_size": patch_size,
        "label_offset": label_offset,
        "mean": mean,
        "std": std,
        "augment": False,
        "pad_mode": str(train_cfg.get("pad_mode", "edge")),
    }

    def make_ds(indices: np.ndarray):
        kw = dict(common)
        kw["indices"] = indices
        kw = _filter_kwargs_by_signature(HSIPatchDataset, kw)
        return HSIPatchDataset(**kw)

    ds_va = make_ds(va_idx)
    ds_te = make_ds(te_idx)

    dl_va = DataLoader(ds_va, batch_size=int(args.batch_size), shuffle=False,
                       num_workers=int(args.num_workers), pin_memory=True, drop_last=False,
                       persistent_workers=False)
    dl_te = DataLoader(ds_te, batch_size=int(args.batch_size), shuffle=False,
                       num_workers=int(args.num_workers), pin_memory=True, drop_last=False,
                       persistent_workers=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_raw_bands = int(cube.shape[-1])
    model = _make_model(mcfg, num_classes=num_classes, raw_bands=model_raw_bands).to(device)
    num_params_total = int(sum(p.numel() for p in model.parameters()))
    num_params_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    ck = _torch_load_compat(str(ckpt_path), map_location="cpu")
    state = ck.get(args.ckpt_key, ck)
    model.load_state_dict(state, strict=True)

    use_amp = bool(args.amp and device.type == "cuda")
    log_interval = max(0, int(args.log_interval))
    print(
        f"[run] mode=eval model=CenterTargetMamba dataset={dataset} seed={int(args.seed)} "
        f"device={device.type} amp={use_amp} patch={patch_size}"
    )
    print(
        f"[run] split={len(ds_va)}/{len(ds_te)}(val/te) "
        f"batch={int(args.batch_size)} workers={int(args.num_workers)} ckpt_key={args.ckpt_key}"
    )
    val_m = engine_evaluate(
        model=model,
        dl=dl_va,
        device=device,
        num_classes=num_classes,
        use_amp=use_amp,
        patch_size=patch_size,
        log_prefix="eval_val",
        log_interval=log_interval,
    )
    test_m = engine_evaluate(
        model=model,
        dl=dl_te,
        device=device,
        num_classes=num_classes,
        use_amp=use_amp,
        patch_size=patch_size,
        log_prefix="eval_test",
        log_interval=log_interval,
    )

    train_cfg_path = Path(str(args.train_cfg)).resolve() if str(args.train_cfg).strip() else None
    split_json_path = Path(args.split_json)

    out = {
        "VAL": val_m,
        "TEST": test_m,
        "time_sec": float(time.time() - t0),
        "meta": {
            "dataset": dataset,
            "split_json": str(Path(args.split_json)),
            "split_tag": str(args.split_tag),
            "seed": int(args.seed),
            "ckpt": str(ckpt_path),
            "ckpt_key": str(args.ckpt_key),
            "label_offset": int(label_offset),
            "num_classes": int(num_classes),
            "patch_size": int(patch_size),
            "dataset_cfg": str(Path(args.dataset_cfg)),
            "model_cfg": str(Path(args.model_cfg)),
            "dataset_cfg_sha1": _sha1_file(args.dataset_cfg),
            "model_cfg_sha1": _sha1_file(args.model_cfg),
            "train_cfg": str(train_cfg_path) if train_cfg_path is not None else "",
            "train_cfg_sha1": _sha1_file(train_cfg_path) if train_cfg_path is not None else "",
            "split_json_sha1": _sha1_file(split_json_path),
            "norm_path": str(norm_path) if norm_path.exists() else "computed_from_train",
            "spectral_preprocess_path": str(spectral_preprocess_path) if spectral_preprocess_path.exists() else "",
            "spectral_preprocess_mode": str(spectral_state.get("mode", "none")),
            "spectral_raw_bands": int(spectral_state.get("raw_bands", cube.shape[-1])),
            "spectral_out_bands": int(spectral_state.get("out_bands", cube.shape[-1])),
            "num_params_total": num_params_total,
            "num_params_trainable": num_params_trainable,
        },
    }

    out_p = Path(args.out)
    ensure_dir(out_p.parent)
    save_json(out_p, out)

    print(
        f"[done] out={out_p} "
        f"val_OA={val_m['OA']:.6f} val_AA={val_m['AA']:.6f} val_Kappa={val_m['Kappa']:.6f} "
        f"test_OA={test_m['OA']:.6f} test_AA={test_m['AA']:.6f} test_Kappa={test_m['Kappa']:.6f} "
        f"time_sec={out['time_sec']:.2f}"
    )


if __name__ == "__main__":
    main()
