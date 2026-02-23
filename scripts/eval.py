#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation script (protocol-safe, NO TTA)."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from hsi3d.utils.io import load_yaml, load_json, ensure_dir, save_json
from hsi3d.utils.seed import set_global_seed
from hsi3d.data.hsi_dataset import HSIPatchDataset, compute_train_norm
from hsi3d.models import VSSM3DConfig, VSSM3DMoE


def _metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    cm = cm.astype(np.float64, copy=False)
    n = cm.sum()
    if n <= 0:
        return {"OA": 0.0, "AA": 0.0, "Kappa": 0.0}

    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred = cm.sum(axis=0)

    oa = float(tp.sum() / n)

    valid = support > 0
    acc_per_class = np.zeros_like(support)
    acc_per_class[valid] = tp[valid] / support[valid]
    aa = float(acc_per_class[valid].mean()) if valid.any() else 0.0

    pe = float((support * pred).sum() / (n * n))
    kappa = float((oa - pe) / max(1e-12, 1.0 - pe))
    return {"OA": oa, "AA": aa, "Kappa": kappa}


_ALPHA_MIN = 1e-6
_ALPHA_MAX = 1e4
_EPS = 1e-12


def _sanitize_alpha(alpha: torch.Tensor) -> torch.Tensor:
    a = alpha.float()
    a = torch.nan_to_num(a, nan=1.0, posinf=_ALPHA_MAX, neginf=1.0)
    a = torch.clamp(a, min=_ALPHA_MIN, max=_ALPHA_MAX)
    return a


def _alpha_to_prob(alpha: torch.Tensor) -> torch.Tensor:
    a = _sanitize_alpha(alpha)
    denom = a.sum(dim=1, keepdim=True).clamp_min(_EPS)
    return a / denom


@torch.no_grad()
def _eval_metrics(model: torch.nn.Module, dl: DataLoader, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, x_spec, y in dl:
        x = x.to(device, non_blocking=True)
        x_spec = x_spec.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x, x_spec)
        prob = _alpha_to_prob(out)
        pred = torch.argmax(prob, dim=1)

        yt = y.detach().cpu().numpy().astype(np.int64, copy=False)
        yp = pred.detach().cpu().numpy().astype(np.int64, copy=False)
        np.add.at(cm, (yt, yp), 1)

    return _metrics_from_cm(cm)


def _filter_kwargs_by_signature(fn_or_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn_or_cls)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _make_model(model_cfg: Dict[str, Any], num_classes: int, raw_bands: int) -> torch.nn.Module:
    mcfg = dict(model_cfg or {})
    cfg_kwargs = {
        "num_classes": int(num_classes),
        "raw_bands": int(raw_bands),
        "patch_size": int(mcfg.get("patch_size", 15)),
        "dropout": float(mcfg.get("dropout", 0.15)),
        "stages": tuple(mcfg.get("stages", [2, 2, 4])),
        "stage_dims": tuple(mcfg.get("stage_dims", [64, 96, 128])),
        "spec_groups": int(mcfg.get("spec_groups", 8)),
        "spec_layers": int(mcfg.get("spec_layers", 3)),
        "spec_hidden": int(mcfg.get("spec_hidden", mcfg.get("d_model", 96))),
        "moe_experts": int(mcfg.get("moe_experts", 3)),
        "moe_topk": int(mcfg.get("moe_topk", 1)),
        "head": str(mcfg.get("head", "evidential")),
    }
    cfg_kwargs = _filter_kwargs_by_signature(VSSM3DConfig, cfg_kwargs)
    cfg = VSSM3DConfig(**cfg_kwargs)
    return VSSM3DMoE(cfg)


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
    p.add_argument("--out", required=True)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    set_global_seed(int(args.seed))

    dcfg = load_yaml(args.dataset_cfg)
    mcfg = load_yaml(args.model_cfg)
    split = load_json(args.split_json)

    dataset = str(dcfg.get("dataset", split.get("dataset", ""))).strip()
    if not dataset:
        raise SystemExit("[ERROR] dataset name not found in dataset_cfg/split_json.")

    raw_dir = Path(args.data_root) / "processed" / dataset / "raw"
    cube = np.load(raw_dir / "cube.npy")
    gt = np.load(raw_dir / "gt.npy")

    label_offset = int(split.get("label_offset", dcfg.get("label_offset", 1)))
    num_classes = int(gt.max()) - label_offset + 1
    if num_classes <= 1:
        raise SystemExit(f"[ERROR] bad num_classes={num_classes}, label_offset={label_offset}, gt.max={int(gt.max())}")

    tr_idx = np.asarray(split.get("train_indices", split.get("train", [])), dtype=np.int64)
    va_idx = np.asarray(split.get("val_indices", split.get("val", [])), dtype=np.int64)
    te_idx = np.asarray(split.get("test_indices", split.get("test", [])), dtype=np.int64)

    patch_size = int(mcfg.get("patch_size", dcfg.get("patch_size", 15)))

    ckpt_path = Path(args.ckpt)
    out_dir = ckpt_path.parent.parent
    norm_path = out_dir / "meta" / "norm_stats.npz"
    if norm_path.exists():
        z = np.load(norm_path)
        mean = z["mean"].astype(np.float32)
        std = z["std"].astype(np.float32)
    else:
        mean, std = compute_train_norm(cube, tr_idx)

    common = {
        "cube": cube,
        "gt": gt,
        "patch_size": patch_size,
        "label_offset": label_offset,
        "mean": mean,
        "std": std,
        "augment": False,
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
                       persistent_workers=(int(args.num_workers) > 0))
    dl_te = DataLoader(ds_te, batch_size=int(args.batch_size), shuffle=False,
                       num_workers=int(args.num_workers), pin_memory=True, drop_last=False,
                       persistent_workers=(int(args.num_workers) > 0))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _make_model(mcfg, num_classes=num_classes, raw_bands=cube.shape[-1]).to(device)

    ck = torch.load(str(ckpt_path), map_location="cpu")
    state = ck.get(args.ckpt_key, ck)
    model.load_state_dict(state, strict=True)

    use_amp = bool(args.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
    else:
        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

    with autocast_ctx:
        val_m = _eval_metrics(model, dl_va, device, num_classes)
        test_m = _eval_metrics(model, dl_te, device, num_classes)

    out = {
        "val": val_m,
        "test": test_m,
        "meta": {
            "dataset": dataset,
            "split_json": str(Path(args.split_json)),
            "ckpt": str(ckpt_path),
            "ckpt_key": str(args.ckpt_key),
            "label_offset": int(label_offset),
            "num_classes": int(num_classes),
            "patch_size": int(patch_size),
            "norm_path": str(norm_path) if norm_path.exists() else "computed_from_train",
            "tta": False,
        },
    }

    out_p = Path(args.out)
    ensure_dir(out_p.parent)
    save_json(out_p, out)

    print(
        f"[eval] dataset={dataset} seed={args.seed} "
        f"VAL(OA/AA/Kappa)={val_m['OA']:.4f}/{val_m['AA']:.4f}/{val_m['Kappa']:.4f} "
        f"TEST(OA/AA/Kappa)={test_m['OA']:.4f}/{test_m['AA']:.4f}/{test_m['Kappa']:.4f} "
        f"-> {out_p}"
    )


if __name__ == "__main__":
    main()
