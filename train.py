#!/usr/bin/env python3
"""Training entry point for TriScanMamba."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import yaml

from utils.hsi_dataset import HSIPatchDataset, compute_train_norm
from utils.lr_schedulers import WarmupCosine
from utils.engine import train_one_epoch, evaluate
from models.triscan_mamba import VSSM3DConfig, VSSM3DModel


def _load_yaml(p: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


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


def _torch_load_compat(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
    model = VSSM3DModel(cfg).to(device)
    return model


def _resolve_need_x_spec(train_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> bool:
    _ = model_cfg
    return bool(train_cfg.get("need_x_spec", False))


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

    dataset_cfg = _load_yaml(Path(args.dataset_cfg))
    model_cfg = _load_yaml(Path(args.model_cfg))
    train_cfg = _load_yaml(Path(args.train_cfg))
    split = _load_json(Path(args.split_json))

    deterministic = bool(train_cfg.get("deterministic", True))
    _set_seed(int(args.seed), deterministic=deterministic)

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
    append_coords = bool(model_cfg.get("append_coords", train_cfg.get("append_coords", False)))

    tr_indices = np.asarray(split["train_indices"], dtype=np.int64)
    va_indices = np.asarray(split["val_indices"], dtype=np.int64)
    te_indices = np.asarray(split["test_indices"], dtype=np.int64)

    label_offset = int(split.get("label_offset", dataset_cfg.get("label_offset", 1)))
    num_classes = int(split.get("num_classes", dataset_cfg.get("num_classes", int(gt.max()))))

    mean, std = compute_train_norm(cube, tr_indices)
    np.savez(out_dir / "meta" / "norm_stats.npz", mean=mean, std=std)

    patch_size = int(model_cfg.get("patch_size", 15))
    need_x_spec = _resolve_need_x_spec(train_cfg, model_cfg)

    ds_train_kwargs = dict(
        cube=cube, gt=gt, indices=tr_indices, patch_size=patch_size, mean=mean, std=std,
        label_offset=label_offset,
        augment=bool(train_cfg.get("augment", False)),
        return_x_spec=need_x_spec,
        append_coords=append_coords,
        spec_dropout_p=float(train_cfg.get("spec_dropout_p", 0.0)),
        spec_dropout_ratio=float(train_cfg.get("spec_dropout_ratio", 0.0)),
        noise_std=float(train_cfg.get("noise_std", 0.0)),
    )
    ds_eval_kwargs = dict(
        cube=cube, gt=gt, patch_size=patch_size, mean=mean, std=std, label_offset=label_offset, augment=False,
        return_x_spec=need_x_spec,
        append_coords=append_coords,
    )

    ds_tr = HSIPatchDataset(**_filter_kwargs(HSIPatchDataset.__init__, ds_train_kwargs))
    ds_va = HSIPatchDataset(**_filter_kwargs(HSIPatchDataset.__init__, dict(ds_eval_kwargs, indices=va_indices)))
    ds_te = HSIPatchDataset(**_filter_kwargs(HSIPatchDataset.__init__, dict(ds_eval_kwargs, indices=te_indices)))

    batch_size = int(train_cfg.get("batch_size", 16))
    eval_batch_size = int(train_cfg.get("eval_batch_size", 512))
    drop_last = bool(train_cfg.get("drop_last", False))
    num_workers = int(args.num_workers)
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    persistent_workers = bool(num_workers > 0)
    eval_num_workers = max(0, int(train_cfg.get("eval_num_workers", min(num_workers, 4))))
    eval_prefetch_factor = int(train_cfg.get("eval_prefetch_factor", prefetch_factor))

    balanced_sampler = bool(train_cfg.get("balanced_sampler", False))
    balanced_power = float(train_cfg.get("balanced_power", 0.5))
    sampler = _make_balanced_sampler(gt, tr_indices, label_offset=label_offset, power=balanced_power) if balanced_sampler else None

    dl_tr_common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0 and prefetch_factor > 0:
        dl_tr_common["prefetch_factor"] = int(prefetch_factor)

    dl_eval_common = dict(
        num_workers=eval_num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    if eval_num_workers > 0 and eval_prefetch_factor > 0:
        dl_eval_common["prefetch_factor"] = int(eval_prefetch_factor)

    dl_tr = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
        drop_last=drop_last, **dl_tr_common
    )
    dl_va = DataLoader(ds_va, batch_size=eval_batch_size, shuffle=False, drop_last=False, **dl_eval_common)
    dl_te = DataLoader(ds_te, batch_size=eval_batch_size, shuffle=False, drop_last=False, **dl_eval_common)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = _maybe_scaler(use_amp)
    model_raw_bands = int(raw_bands + (2 if append_coords else 0))
    model = _build_model(
        model_cfg,
        num_classes=num_classes,
        raw_bands=model_raw_bands,
        device=device,
        patch_size=patch_size,
    )
    ema_decay = float(train_cfg.get("ema_decay", 0.0))
    ema_start_epoch = int(train_cfg.get("ema_start_epoch", 0))
    use_ema_eval = bool(train_cfg.get("use_ema_eval", True))
    if ema_decay < 0.0:
        ema_decay = 0.0
    if ema_decay >= 1.0:
        ema_decay = 0.9999
    ema_model = None
    if ema_decay > 0.0:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    def _ema_update(dst: torch.nn.Module, src: torch.nn.Module, decay: float) -> None:
        with torch.no_grad():
            d_params = dict(dst.named_parameters())
            s_params = dict(src.named_parameters())
            for n, dp in d_params.items():
                sp = s_params[n]
                dp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)
            d_bufs = dict(dst.named_buffers())
            s_bufs = dict(src.named_buffers())
            for n, db in d_bufs.items():
                db.data.copy_(s_bufs[n].data)

    print(
        f"[runtime] device={device} amp={use_amp} deterministic={deterministic} "
        f"train_workers={num_workers} eval_workers={eval_num_workers} "
        f"need_x_spec={need_x_spec} append_coords={append_coords} "
        f"ema_decay={ema_decay:.6f} ema_start={ema_start_epoch}"
    )

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

    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))

    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    conf_penalty = float(train_cfg.get("conf_penalty", 0.0))
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    mixup_prob = float(train_cfg.get("mixup_prob", 0.0))

    pseudo_label_enable = bool(train_cfg.get("pseudo_label_enable", False))
    pseudo_conf_thr = float(train_cfg.get("pseudo_conf_thr", 0.98))
    pseudo_max_per_class = int(train_cfg.get("pseudo_max_per_class", 1500))
    pseudo_epochs = int(train_cfg.get("pseudo_epochs", 0))
    pseudo_lr_scale = float(train_cfg.get("pseudo_lr_scale", 0.2))
    pseudo_mixup_alpha = float(train_cfg.get("pseudo_mixup_alpha", 0.0))
    pseudo_mixup_prob = float(train_cfg.get("pseudo_mixup_prob", 0.0))
    pseudo_spec_dropout_p = float(train_cfg.get("pseudo_spec_dropout_p", 0.0))
    pseudo_spec_dropout_ratio = float(train_cfg.get("pseudo_spec_dropout_ratio", 0.0))

    focal_gamma = float(train_cfg.get("focal_gamma", 0.0))
    aug_noise_std = float(train_cfg.get("aug_noise_std", 0.0))
    # Keep main-experiment defaults fully enabled; use explicit overrides when needed.
    device_spec_dropout_p = float(train_cfg.get("device_spec_dropout_p", train_cfg.get("spec_dropout_p", 0.0)))
    device_spec_dropout_ratio = float(
        train_cfg.get("device_spec_dropout_ratio", train_cfg.get("spec_dropout_ratio", 0.0))
    )

    early_stop = bool(train_cfg.get("early_stop", True))
    patience = int(train_cfg.get("early_stop_patience", 70))
    min_epochs = int(train_cfg.get("min_epochs", 85))
    min_delta = float(train_cfg.get("early_stop_min_delta", 1.0e-4))
    perfect_score_thr = float(train_cfg.get("perfect_score_thr", 1.1))
    perfect_score_patience = int(train_cfg.get("perfect_score_patience", 0))

    select_metric = str(train_cfg.get("select_metric", "kappa")).lower()
    smooth_k = int(train_cfg.get("val_smooth_k", 15))
    eval_every = max(1, int(train_cfg.get("eval_every", 1)))
    eval_on_first = bool(train_cfg.get("eval_on_first", True))
    final_eval_amp = bool(train_cfg.get("final_eval_amp", True)) and use_amp
    final_eval_num_workers = max(0, int(train_cfg.get("final_eval_num_workers", eval_num_workers)))
    final_eval_prefetch_factor = int(train_cfg.get("final_eval_prefetch_factor", eval_prefetch_factor))
    final_eval_log_interval = max(0, int(train_cfg.get("final_eval_log_interval", 25)))

    best_ep = -1
    best_smooth = -1e18
    no_improve = 0
    perfect_hits = 0
    hist = []

    t0 = time.time()
    for ep in range(max_epochs):
        ep_t0 = time.time()
        train_kwargs = dict(
            model=model,
            dl=dl_tr,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp, scaler=scaler, grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
            label_smoothing=label_smoothing, conf_penalty=conf_penalty,
            mixup_alpha=mixup_alpha, mixup_prob=mixup_prob,
            focal_gamma=focal_gamma,
            aug_noise_std=aug_noise_std,
            patch_size=patch_size,
            spec_dropout_p=device_spec_dropout_p,
            spec_dropout_ratio=device_spec_dropout_ratio,
            need_x_spec=need_x_spec,
        )
        loss_out = train_one_epoch(**_filter_kwargs(train_one_epoch, train_kwargs))
        loss = _unwrap_scalar(loss_out)

        if ema_model is not None and ep >= ema_start_epoch:
            _ema_update(ema_model, model, ema_decay)

        sched.step(ep)
        lr_now = float(optimizer.param_groups[0]["lr"])

        dt_ep = time.time() - ep_t0
        dt_total = time.time() - t0
        do_eval = (ep == 0 and eval_on_first) or (((ep + 1) % eval_every) == 0) or (ep == (max_epochs - 1))
        if do_eval:
            eval_model = model
            if ema_model is not None and use_ema_eval and ep >= ema_start_epoch:
                eval_model = ema_model

            eval_kwargs = dict(
                model=eval_model,
                dl=dl_va,
                device=device,
                num_classes=num_classes,
                use_amp=use_amp,
                patch_size=patch_size,
                need_x_spec=need_x_spec,
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
            if float(score) >= float(perfect_score_thr):
                perfect_hits += 1
            else:
                perfect_hits = 0

            print(
                f"[ep {ep:04d}] loss={_fmt(loss,'0.6f')} lr={lr_now:.2e} | "
                f"VAL OA={_fmt(oa,'0.4f')} AA={_fmt(aa,'0.4f')} Kappa={_fmt(kp,'0.4f')} "
                f"score={_fmt(score,'0.4f')} smooth{k}={_fmt(smooth,'0.4f')} "
                f"time_ep={dt_ep:.2f}s total={dt_total:.1f}s"
            )

            improved = smooth > best_smooth + max(1.0e-12, float(min_delta))
            if improved:
                best_smooth = smooth
                best_ep = ep
                no_improve = 0
                ckpt = {
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict() if ema_model is not None else None,
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
                        "ema": bool(ema_model is not None and use_ema_eval and ep >= ema_start_epoch),
                        "ema_decay": float(ema_decay),
                    },
                }
                torch.save(ckpt, out_dir / "checkpoints" / "best.pt")
            else:
                no_improve += 1

            if (
                do_eval
                and early_stop
                and ep >= min_epochs
                and perfect_score_patience > 0
                and perfect_hits >= perfect_score_patience
            ):
                print(
                    f"[early_stop] reached score>={perfect_score_thr:.4f} for "
                    f"{perfect_hits} evals (best_ep={best_ep})"
                )
                break
        else:
            print(
                f"[ep {ep:04d}] loss={_fmt(loss,'0.6f')} lr={lr_now:.2e} | "
                f"VAL skipped (eval_every={eval_every}) time_ep={dt_ep:.2f}s total={dt_total:.1f}s"
            )

        if do_eval and early_stop and ep >= min_epochs and no_improve >= patience:
            print(f"[early_stop] no improve for {no_improve} epochs (best_ep={best_ep})")
            break

    if pseudo_label_enable and pseudo_epochs > 0:
        ckpt_sup = _torch_load_compat(out_dir / "checkpoints" / "best.pt", map_location="cpu")
        sup_key = "model"
        if bool(use_ema_eval) and isinstance(ckpt_sup.get("ema_model", None), dict):
            sup_key = "ema_model"
        model.load_state_dict(ckpt_sup[sup_key])
        model.to(device)
        model.eval()

        infer_workers = max(0, int(train_cfg.get("pseudo_num_workers", eval_num_workers)))
        infer_prefetch = int(train_cfg.get("pseudo_prefetch_factor", eval_prefetch_factor))
        dl_pseudo_kw = dict(
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=infer_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        if infer_workers > 0 and infer_prefetch > 0:
            dl_pseudo_kw["prefetch_factor"] = int(infer_prefetch)
        dl_pseudo = DataLoader(ds_te, **dl_pseudo_kw)

        pred_list = []
        conf_list = []
        with torch.no_grad():
            for batch in dl_pseudo:
                x, x_spec, _ = batch
                x = x.to(device, non_blocking=True)
                if need_x_spec:
                    x_spec = x_spec.to(device, non_blocking=True)
                else:
                    x_spec = None
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    out = model(x, x_spec) if x_spec is not None else model(x)
                prob = torch.softmax(out, dim=1)
                conf, pred = torch.max(prob, dim=1)
                pred_list.append(pred.detach().cpu().numpy().astype(np.int64, copy=False))
                conf_list.append(conf.detach().cpu().numpy().astype(np.float32, copy=False))

        if pred_list:
            pred_all = np.concatenate(pred_list, axis=0)
            conf_all = np.concatenate(conf_list, axis=0)
        else:
            pred_all = np.empty((0,), dtype=np.int64)
            conf_all = np.empty((0,), dtype=np.float32)

        picked_pos = []
        if pred_all.size > 0:
            for cls in range(num_classes):
                m = (pred_all == cls) & (conf_all >= float(pseudo_conf_thr))
                pos = np.where(m)[0]
                if pos.size == 0:
                    continue
                ord_idx = np.argsort(-conf_all[pos])
                keep = pos[ord_idx[: max(0, int(pseudo_max_per_class))]]
                picked_pos.append(keep)
        picked_pos = np.concatenate(picked_pos, axis=0).astype(np.int64, copy=False) if picked_pos else np.empty((0,), dtype=np.int64)

        if picked_pos.size > 0:
            pseudo_indices = te_indices[picked_pos]
            pseudo_pred = pred_all[picked_pos]

            pseudo_gt = np.asarray(gt, dtype=np.int64).copy()
            rr = (pseudo_indices // W).astype(np.int64)
            cc = (pseudo_indices % W).astype(np.int64)
            pseudo_gt[rr, cc] = pseudo_pred + int(label_offset)

            ds_ps = HSIPatchDataset(
                **_filter_kwargs(
                    HSIPatchDataset.__init__,
                    dict(
                        cube=cube,
                        gt=pseudo_gt,
                        indices=pseudo_indices,
                        patch_size=patch_size,
                        mean=mean,
                        std=std,
                        label_offset=label_offset,
                        augment=True,
                        return_x_spec=need_x_spec,
                        spec_dropout_p=float(pseudo_spec_dropout_p),
                        spec_dropout_ratio=float(pseudo_spec_dropout_ratio),
                        noise_std=float(train_cfg.get("pseudo_noise_std", train_cfg.get("noise_std", 0.0))),
                    ),
                )
            )

            ds_mix = ConcatDataset([ds_tr, ds_ps])
            dl_ps = DataLoader(
                ds_mix,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last,
                **dl_tr_common,
            )

            lr_ps = float(max(min_lr, lr * max(1.0e-4, pseudo_lr_scale)))
            opt_ps = torch.optim.AdamW(model.parameters(), lr=lr_ps, weight_decay=wd, betas=betas)

            best_ps_score = -1.0e18
            best_ps_state = None
            print(
                f"[pseudo] selected={int(picked_pos.size)} conf_thr={pseudo_conf_thr:.3f} "
                f"max_per_class={pseudo_max_per_class} epochs={pseudo_epochs} lr={lr_ps:.2e}"
            )
            for pep in range(max(0, int(pseudo_epochs))):
                loss_ps = train_one_epoch(
                    **_filter_kwargs(
                        train_one_epoch,
                        dict(
                            model=model,
                            dl=dl_ps,
                            optimizer=opt_ps,
                            device=device,
                            use_amp=use_amp,
                            scaler=None,
                            grad_clip=grad_clip,
                            grad_accum_steps=grad_accum_steps,
                            label_smoothing=label_smoothing,
                            conf_penalty=conf_penalty,
                            mixup_alpha=pseudo_mixup_alpha,
                            mixup_prob=pseudo_mixup_prob,
                            focal_gamma=focal_gamma,
                            aug_noise_std=aug_noise_std,
                            patch_size=patch_size,
                            spec_dropout_p=pseudo_spec_dropout_p,
                            spec_dropout_ratio=pseudo_spec_dropout_ratio,
                            need_x_spec=need_x_spec,
                        ),
                    )
                )
                val_ps = evaluate(
                    **_filter_kwargs(
                        evaluate,
                        dict(
                            model=model,
                            dl=dl_va,
                            device=device,
                            num_classes=num_classes,
                            use_amp=use_amp,
                            patch_size=patch_size,
                            need_x_spec=need_x_spec,
                        ),
                    )
                )
                oa_ps = _get_metric(val_ps, "OA")
                aa_ps = _get_metric(val_ps, "AA")
                kp_ps = _get_metric(val_ps, "Kappa")
                if select_metric == "oa":
                    score_ps = oa_ps
                elif select_metric == "aa":
                    score_ps = aa_ps
                else:
                    score_ps = kp_ps
                print(
                    f"[pseudo ep {pep:03d}] loss={_fmt(loss_ps,'0.6f')} "
                    f"VAL OA={_fmt(oa_ps,'0.4f')} AA={_fmt(aa_ps,'0.4f')} "
                    f"Kappa={_fmt(kp_ps,'0.4f')} score={_fmt(score_ps,'0.4f')}"
                )
                if float(score_ps) > float(best_ps_score):
                    best_ps_score = float(score_ps)
                    best_ps_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if best_ps_state is not None:
                model.load_state_dict(best_ps_state, strict=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema_model": None,
                        "optimizer": optimizer.state_dict(),
                        "opt": optimizer.state_dict(),
                        "sched": sched.state_dict(),
                        "epoch": int(best_ep),
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
                            "pseudo_label_enable": True,
                            "pseudo_selected": int(picked_pos.size),
                            "pseudo_conf_thr": float(pseudo_conf_thr),
                        },
                    },
                    out_dir / "checkpoints" / "best.pt",
                )
        else:
            print(
                f"[pseudo] no pseudo labels selected (conf_thr={pseudo_conf_thr:.3f}, "
                f"max_per_class={pseudo_max_per_class}); skip pseudo fine-tune"
            )

    print("[final_eval] loading best checkpoint...")
    ckpt = _torch_load_compat(out_dir / "checkpoints" / "best.pt", map_location="cpu")
    final_key = "model"
    if bool(use_ema_eval) and isinstance(ckpt.get("ema_model", None), dict):
        final_key = "ema_model"
    model.load_state_dict(ckpt[final_key])
    model.to(device)
    model.eval()
    print(f"[final_eval] using checkpoint weights: {final_key}")

    dl_final_common = dict(
        num_workers=final_eval_num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    if final_eval_num_workers > 0 and final_eval_prefetch_factor > 0:
        dl_final_common["prefetch_factor"] = int(final_eval_prefetch_factor)
    dl_va_final = DataLoader(ds_va, batch_size=eval_batch_size, shuffle=False, drop_last=False, **dl_final_common)
    dl_te_final = DataLoader(ds_te, batch_size=eval_batch_size, shuffle=False, drop_last=False, **dl_final_common)

    print("[final_eval] evaluating VAL...")
    print(
        f"[final_eval] VAL samples={len(ds_va)} batch={eval_batch_size} "
        f"workers={final_eval_num_workers} amp={final_eval_amp}"
    )
    val_best = evaluate(
        **_filter_kwargs(
            evaluate,
            dict(
                model=model,
                dl=dl_va_final,
                device=device,
                num_classes=num_classes,
                use_amp=final_eval_amp,
                patch_size=patch_size,
                log_prefix="final_val",
                log_interval=final_eval_log_interval,
                need_x_spec=need_x_spec,
            ),
        )
    )
    print("[final_eval] evaluating TEST...")
    print(
        f"[final_eval] TEST samples={len(ds_te)} batch={eval_batch_size} "
        f"workers={final_eval_num_workers} amp={final_eval_amp}"
    )
    te_best = evaluate(
        **_filter_kwargs(
            evaluate,
            dict(
                model=model,
                dl=dl_te_final,
                device=device,
                num_classes=num_classes,
                use_amp=final_eval_amp,
                patch_size=patch_size,
                log_prefix="final_test",
                log_interval=final_eval_log_interval,
                need_x_spec=need_x_spec,
            ),
        )
    )

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
