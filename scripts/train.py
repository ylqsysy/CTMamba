#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure "src" is importable
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.exists():
    _sys.path.insert(0, str(_SRC))

import yaml  # PyYAML

from hsi3d.data.hsi_dataset import HSIPatchDataset, compute_train_norm
from hsi3d.training.lr_schedulers import WarmupCosine
from hsi3d.training.engine import train_one_epoch, evaluate
from hsi3d.models.vssm3d_moe import VSSM3DConfig, VSSM3DMoE


def _load_yaml(p: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _filter_kwargs(fn, kw: Dict[str, Any]) -> Dict[str, Any]:
    import inspect
    sig = inspect.signature(fn)
    return {k: v for k, v in kw.items() if k in sig.parameters}


def _unwrap_scalar(x: Any) -> float:
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    if torch.is_tensor(x):
        try:
            return float(x.detach().item())
        except Exception:
            return float("nan")
    if isinstance(x, dict):
        for k in ("loss", "total_loss", "train_loss", "value", "mean"):
            if k in x:
                return _unwrap_scalar(x[k])
        if len(x):
            return _unwrap_scalar(next(iter(x.values())))
        return float("nan")
    if isinstance(x, (list, tuple)) and len(x):
        return _unwrap_scalar(x[0])
    return float("nan")


def _get_metric(d: Any, key: str) -> float:
    if isinstance(d, dict):
        if key in d:
            return _unwrap_scalar(d[key])
        m = d.get("metrics")
        if isinstance(m, dict) and key in m:
            return _unwrap_scalar(m[key])
    return float("nan")


def _fmt(x: Any, spec: str) -> str:
    v = _unwrap_scalar(x)
    if math.isnan(v) or math.isinf(v):
        return "nan"
    return format(v, spec)


def _make_balanced_sampler(gt: np.ndarray, indices: np.ndarray, *, label_offset: int, power: float) -> WeightedRandomSampler:
    H, W = gt.shape
    r = (indices // W).astype(np.int64)
    c = (indices % W).astype(np.int64)
    y = gt[r, c].astype(np.int64) - int(label_offset)
    y = np.clip(y, 0, None)
    n_cls = int(y.max()) + 1 if y.size else 1
    counts = np.bincount(y, minlength=n_cls).astype(np.float64)
    counts[counts <= 0] = 1.0
    w_cls = 1.0 / np.power(counts, float(power))
    weights = w_cls[y]
    weights = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(indices), replacement=True)


def _maybe_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def _build_model(
    model_cfg: Dict[str, Any],
    *,
    num_classes: int,
    raw_bands: int,
    device: torch.device,
    patch_size: int | None = None,
    **_: Any,
) -> torch.nn.Module:
    cfg_in = dict(model_cfg)

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

    if patch_size is not None:
        cfg_in["patch_size"] = int(patch_size)

    allowed = set(getattr(VSSM3DConfig, "__annotations__", {}).keys())
    cfg_in = {k: v for k, v in cfg_in.items() if k in allowed}

    cfg = VSSM3DConfig(**cfg_in)
    model = VSSM3DMoE(cfg).to(device)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", type=str, required=True)
    ap.add_argument("--model_cfg", type=str, required=True)
    ap.add_argument("--train_cfg", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    _set_seed(int(args.seed))

    dataset_cfg = _load_yaml(Path(args.dataset_cfg))
    model_cfg = _load_yaml(Path(args.model_cfg))
    train_cfg = _load_yaml(Path(args.train_cfg))
    split = _load_json(Path(args.split_json))

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "checkpoints")
    _ensure_dir(out_dir / "meta")

    data_root = Path(args.data_root)
    ds_name = str(dataset_cfg.get("name", dataset_cfg.get("dataset", "hanchuan")))
    proc_dir = data_root / "processed" / ds_name / "raw"

    cube = np.load(proc_dir / "cube.npy")
    gt = np.load(proc_dir / "gt.npy")
    if gt.ndim != 2:
        raise ValueError(f"gt.npy must be (H,W), got {gt.shape}")
    H, W = gt.shape

    if cube.ndim != 3:
        raise ValueError(f"cube.npy must be 3D, got {cube.shape}")
    if cube.shape[:2] == (H, W):
        pass
    elif cube.shape[-2:] == (H, W):
        cube = np.transpose(cube, (1, 2, 0))
    else:
        raise ValueError(f"cube.npy shape {cube.shape} does not match gt shape {gt.shape}")

    raw_bands = int(cube.shape[-1])

    tr_indices = np.asarray(split["train_indices"], dtype=np.int64)
    va_indices = np.asarray(split["val_indices"], dtype=np.int64)
    te_indices = np.asarray(split["test_indices"], dtype=np.int64)

    label_offset = int(split.get("label_offset", dataset_cfg.get("label_offset", 1)))
    num_classes = int(split.get("num_classes", dataset_cfg.get("num_classes", int(gt.max()))))

    mean, std = compute_train_norm(cube, tr_indices)
    np.savez(out_dir / "meta" / "norm_stats.npz", mean=mean, std=std)

    patch_size = int(model_cfg.get("patch_size", 15))

    ds_train_kwargs = dict(
        cube=cube, gt=gt, indices=tr_indices, patch_size=patch_size, mean=mean, std=std,
        label_offset=label_offset,
        augment=bool(train_cfg.get("augment", False)),
        spec_dropout_p=float(train_cfg.get("spec_dropout_p", 0.0)),
        spec_dropout_ratio=float(train_cfg.get("spec_dropout_ratio", 0.0)),
        noise_std=float(train_cfg.get("noise_std", 0.0)),
        spec_jitter_std=float(train_cfg.get("spec_jitter_std", 0.0)),
        spec_shift_p=float(train_cfg.get("spec_shift_p", 0.0)),
        spec_shift_max=int(train_cfg.get("spec_shift_max", 0)),
    )
    ds_eval_kwargs = dict(
        cube=cube, gt=gt, patch_size=patch_size, mean=mean, std=std, label_offset=label_offset, augment=False,
    )

    ds_tr = HSIPatchDataset(**_filter_kwargs(HSIPatchDataset.__init__, ds_train_kwargs))
    ds_va = HSIPatchDataset(**_filter_kwargs(HSIPatchDataset.__init__, dict(ds_eval_kwargs, indices=va_indices)))
    ds_te = HSIPatchDataset(**_filter_kwargs(HSIPatchDataset.__init__, dict(ds_eval_kwargs, indices=te_indices)))

    batch_size = int(train_cfg.get("batch_size", 16))
    eval_batch_size = int(train_cfg.get("eval_batch_size", 512))
    drop_last = bool(train_cfg.get("drop_last", False))
    num_workers = int(args.num_workers)

    balanced_sampler = bool(train_cfg.get("balanced_sampler", False))
    balanced_power = float(train_cfg.get("balanced_power", 0.5))
    sampler = _make_balanced_sampler(gt, tr_indices, label_offset=label_offset, power=balanced_power) if balanced_sampler else None

    dl_tr = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, drop_last=drop_last, pin_memory=True
    )
    dl_va = DataLoader(ds_va, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = _maybe_scaler(use_amp)

    model = _build_model(model_cfg, num_classes=num_classes, raw_bands=raw_bands, device=device, patch_size=patch_size)

    lr = float(train_cfg.get("lr", 1.1e-4))
    wd = float(train_cfg.get("weight_decay", 3.0e-2))
    betas = train_cfg.get("betas", [0.9, 0.999])
    betas = (float(betas[0]), float(betas[1]))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    max_epochs = int(train_cfg.get("max_epochs", 340))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 12))
    min_lr = float(train_cfg.get("min_lr", 1.0e-6))
    sched = WarmupCosine(optimizer, max_epochs=max_epochs, warmup_epochs=warmup_epochs, min_lr=min_lr)

    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 0))
    if steps_per_epoch <= 0:
        steps_per_epoch = len(dl_tr)

    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))

    evi_coeff = float(train_cfg.get("evi_coeff", 1.0))
    evi_anneal_epochs = int(train_cfg.get("evi_anneal_epochs", 130))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    conf_penalty = float(train_cfg.get("conf_penalty", 0.0))
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    mixup_prob = float(train_cfg.get("mixup_prob", 0.0))

    focal_gamma = float(train_cfg.get("focal_gamma", 0.0))
    spec_cutout_prob = float(train_cfg.get("spec_cutout_prob", 0.0))
    spec_cutout_ratio = float(train_cfg.get("spec_cutout_ratio", 0.0))
    spatial_cutout_prob = float(train_cfg.get("spatial_cutout_prob", 0.0))
    spatial_cutout_ratio = float(train_cfg.get("spatial_cutout_ratio", 0.0))
    aug_noise_std = float(train_cfg.get("aug_noise_std", 0.0))

    early_stop = bool(train_cfg.get("early_stop", True))
    patience = int(train_cfg.get("early_stop_patience", 70))
    min_epochs = int(train_cfg.get("min_epochs", 85))

    select_metric = str(train_cfg.get("select_metric", "kappa")).lower()
    smooth_k = int(train_cfg.get("val_smooth_k", 15))

    best_ep = -1
    best_smooth = -1e18
    no_improve = 0
    hist = []

    t0 = time.time()
    for ep in range(max_epochs):
        # Provide multiple aliases for loader/optimizer to match any engine signature
        train_kwargs = dict(
            model=model,
            dl=dl_tr, loader=dl_tr, train_loader=dl_tr, data_loader=dl_tr,
            optimizer=optimizer, opt=optimizer,
            device=device,
            amp=use_amp, scaler=scaler, grad_clip=grad_clip,
            steps_per_epoch=steps_per_epoch, grad_accum_steps=grad_accum_steps,
            evi_coeff=evi_coeff, evi_anneal_epochs=evi_anneal_epochs,
            label_smoothing=label_smoothing, conf_penalty=conf_penalty,
            mixup_alpha=mixup_alpha, mixup_prob=mixup_prob,
            focal_gamma=focal_gamma,
            spec_cutout_prob=spec_cutout_prob, spec_cutout_ratio=spec_cutout_ratio,
            spatial_cutout_prob=spatial_cutout_prob, spatial_cutout_ratio=spatial_cutout_ratio,
            aug_noise_std=aug_noise_std,
            epoch=ep,  # will be ignored if unsupported
        )
        loss_out = train_one_epoch(**_filter_kwargs(train_one_epoch, train_kwargs))
        loss = _unwrap_scalar(loss_out)

        sched.step(ep)
        lr_now = float(optimizer.param_groups[0]["lr"])

        eval_kwargs = dict(
            model=model,
            dl=dl_va, loader=dl_va, val_loader=dl_va, data_loader=dl_va,
            device=device,
            num_classes=num_classes,
            amp=use_amp,
        )
        val_out = evaluate(**_filter_kwargs(evaluate, eval_kwargs))

        oa = _get_metric(val_out, "OA")
        aa = _get_metric(val_out, "AA")
        kp = _get_metric(val_out, "Kappa")

        if select_metric == "oa":
            score = oa
        elif select_metric == "aa":
            score = aa
        else:
            score = kp

        hist.append(float(score))
        k = max(1, smooth_k)
        smooth = float(np.mean(hist[-k:]))

        dt = time.time() - t0
        print(f"[ep {ep:04d}] loss={_fmt(loss,'0.6f')} lr={lr_now:.2e} | VAL OA={_fmt(oa,'0.4f')} AA={_fmt(aa,'0.4f')} Kappa={_fmt(kp,'0.4f')} score={_fmt(score,'0.4f')} smooth{k}={_fmt(smooth,'0.4f')} time={dt:.1f}s")

        improved = smooth > best_smooth + 1e-12
        if improved:
            best_smooth = smooth
            best_ep = ep
            no_improve = 0
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "opt": optimizer.state_dict(),
                "sched": sched.state_dict(),
                "epoch": int(ep),
                "best_smooth": float(best_smooth),
                "meta": {
                    "dataset": ds_name,
                    "split_json": str(Path(args.split_json)),
                    "label_offset": int(label_offset),
                    "num_classes": int(num_classes),
                    "patch_size": int(patch_size),
                    "norm_path": str(out_dir / "meta" / "norm_stats.npz"),
                    "tta": False,
                    "ema": False,
                },
            }
            torch.save(ckpt, out_dir / "checkpoints" / "best.pt")
        else:
            no_improve += 1

        if early_stop and ep >= min_epochs and no_improve >= patience:
            print(f"[early_stop] no improve for {no_improve} epochs (best_ep={best_ep})")
            break

    ckpt = torch.load(out_dir / "checkpoints" / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    val_best = evaluate(**_filter_kwargs(evaluate, dict(model=model, dl=dl_va, loader=dl_va, device=device, num_classes=num_classes, amp=False)))
    te_best = evaluate(**_filter_kwargs(evaluate, dict(model=model, dl=dl_te, loader=dl_te, device=device, num_classes=num_classes, amp=False)))

    result = {
        "best_ep": int(best_ep),
        "VAL": val_best,
        "TEST": te_best,
        "time_sec": float(time.time() - t0),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] out_dir={out_dir} best_ep={best_ep} ckpt=best VAL={val_best} TEST={te_best} time={result['time_sec']:.1f}s")


if __name__ == "__main__":
    main()
