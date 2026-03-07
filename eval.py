#!/usr/bin/env python3
"""Evaluate a trained checkpoint on validation and test splits.

This entry point intentionally performs a single deterministic forward pass
without any test-time augmentation so that metrics are directly comparable
across repeated runs.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Dict, Any
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from utils.io import load_yaml, load_json, ensure_dir, save_json
from utils.seed import set_global_seed
from utils.hsi_dataset import HSIPatchDataset, compute_train_norm
from utils.engine import evaluate as engine_evaluate
from models import VSSM3DConfig, VSSM3DModel


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


def _extract_logits(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
        return out[0]
    if isinstance(out, dict):
        for k in ("logits", "pred", "y", "scores"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        vals = [v for v in out.values() if torch.is_tensor(v)]
        if vals:
            return vals[0]
    raise TypeError(f"Unsupported model output type: {type(out)}")


@torch.no_grad()
def _eval_metrics(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    use_amp: bool = False,
    log_prefix: str = "",
    log_interval: int = 0,
) -> Dict[str, float]:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    try:
        total_steps = len(dl)
    except Exception:
        total_steps = None
    lp = str(log_prefix).strip()
    li = max(0, int(log_interval))

    for it, (x, x_spec, y) in enumerate(dl):
        x = x.to(device, non_blocking=True)
        x_spec = x_spec.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(x, x_spec)
        logits = _extract_logits(out)
        pred = torch.argmax(logits, dim=1)

        yt = y.detach().cpu().numpy().astype(np.int64, copy=False)
        yp = pred.detach().cpu().numpy().astype(np.int64, copy=False)
        np.add.at(cm, (yt, yp), 1)

        if li > 0 and (((it + 1) % li == 0) or (it == 0)):
            done = it + 1
            if total_steps is None:
                p = "?"
            else:
                p = f"{done / max(1, total_steps):.1%}"
            tag = f"[{lp}] " if lp else ""
            denom = str(total_steps) if total_steps is not None else "?"
            print(f"{tag}eval {done}/{denom} ({p})", flush=True)

    return _metrics_from_cm(cm)


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

    allowed = set(getattr(VSSM3DConfig, "__annotations__", {}).keys())
    cfg_kwargs = {k: v for k, v in cfg_in.items() if k in allowed}
    cfg = VSSM3DConfig(**cfg_kwargs)
    return VSSM3DModel(cfg)


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
    append_coords = bool(mcfg.get("append_coords", False))

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
        "append_coords": append_coords,
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
    model_raw_bands = int(cube.shape[-1] + (2 if append_coords else 0))
    model = _make_model(mcfg, num_classes=num_classes, raw_bands=model_raw_bands).to(device)

    ck = _torch_load_compat(str(ckpt_path), map_location="cpu")
    state = ck.get(args.ckpt_key, ck)
    model.load_state_dict(state, strict=True)

    use_amp = bool(args.amp and device.type == "cuda")
    log_interval = max(0, int(args.log_interval))
    print(
        f"[eval] dataset={dataset} seed={args.seed} "
        f"VAL_samples={len(ds_va)} TEST_samples={len(ds_te)} "
        f"batch={int(args.batch_size)} workers={int(args.num_workers)} amp={use_amp}"
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
