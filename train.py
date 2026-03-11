#!/usr/bin/env python3
"""Training entry point for CenterTargetMamba."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import yaml

from utils.hsi_dataset import HSIPatchDataset, compute_train_norm
from utils.hsi_preprocess import fit_and_apply_spectral_preprocess
from utils.lr_schedulers import WarmupCosine
from utils.engine import train_one_epoch, evaluate
from models.ctmamba import CTMambaConfig, CTMamba


def _load_yaml(p: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _set_seed(seed: int, deterministic: bool = False) -> None:
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

    allowed = set(getattr(CTMambaConfig, "__annotations__", {}).keys())
    unknown = sorted([k for k in cfg_in.keys() if k not in allowed])
    if unknown:
        raise ValueError(
            f"Unknown model config keys: {unknown}. "
            f"Allowed keys: {sorted(allowed)}"
        )

    cfg = CTMambaConfig(**cfg_in)
    model = CTMamba(cfg).to(device)
    return model


def _resolve_need_x_spec(train_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> bool:
    _ = model_cfg
    if bool(train_cfg.get("need_x_spec", False)):
        print("[warn] train_cfg.need_x_spec=true is ignored: current model does not use x_spec.", flush=True)
    return False


def _resolve_num_classes(split: Dict[str, Any], dataset_cfg: Dict[str, Any], gt: np.ndarray, label_offset: int) -> int:
    fallback = int(gt.max()) - int(label_offset) + 1
    num_classes = int(split.get("num_classes", dataset_cfg.get("num_classes", fallback)))
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")
    return int(num_classes)


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
    ap.add_argument("--init_ckpt", type=str, default="", help="Optional pretrained checkpoint path.")
    ap.add_argument("--init_ckpt_key", type=str, default="model", help="State-dict key for --init_ckpt.")
    args = ap.parse_args()

    dataset_cfg_path = Path(args.dataset_cfg)
    model_cfg_path = Path(args.model_cfg)
    train_cfg_path = Path(args.train_cfg)
    split_json_path = Path(args.split_json)

    dataset_cfg = _load_yaml(dataset_cfg_path)
    model_cfg = _load_yaml(model_cfg_path)
    train_cfg = _load_yaml(train_cfg_path)
    split = _load_json(split_json_path)

    deterministic = bool(train_cfg.get("deterministic", False))
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

    tr_indices = np.asarray(split["train_indices"], dtype=np.int64)
    va_indices = np.asarray(split["val_indices"], dtype=np.int64)
    te_indices = np.asarray(split["test_indices"], dtype=np.int64)

    spectral_preprocess_path = out_dir / "meta" / "spectral_preprocess.npz"
    cube, spectral_state = fit_and_apply_spectral_preprocess(
        cube,
        tr_indices,
        train_cfg,
        gt_shape=(H, W),
        save_path=spectral_preprocess_path,
    )
    raw_bands = int(cube.shape[-1])
    if str(spectral_state.get("mode", "none")) != "none":
        extra = ""
        evr = np.asarray(spectral_state.get("explained_variance_ratio", []), dtype=np.float32).reshape(-1)
        if evr.size > 0:
            extra = f" explained_var_sum={float(evr.sum()):.4f}"
        print(
            f"[run] spectral_preprocess={spectral_state.get('mode')} "
            f"{int(spectral_state.get('raw_bands', raw_bands))}->{int(spectral_state.get('out_bands', raw_bands))}"
            f"{extra}"
        )

    label_offset = int(split.get("label_offset", dataset_cfg.get("label_offset", 1)))
    num_classes = _resolve_num_classes(split, dataset_cfg, gt, label_offset)

    dataset_cfg_sha1 = _sha1_file(dataset_cfg_path)
    model_cfg_sha1 = _sha1_file(model_cfg_path)
    train_cfg_sha1 = _sha1_file(train_cfg_path)
    split_json_sha1 = _sha1_file(split_json_path)

    norm_mean_global_blend = float(train_cfg.get("norm_mean_global_blend", 0.0))
    norm_std_abs_floor = float(train_cfg.get("norm_std_abs_floor", 1.0e-3))
    mean, std = compute_train_norm(
        cube,
        tr_indices,
        mean_global_blend=norm_mean_global_blend,
        std_abs_floor=norm_std_abs_floor,
    )
    np.savez(out_dir / "meta" / "norm_stats.npz", mean=mean, std=std)

    patch_size = int(model_cfg.get("patch_size", 15))
    need_x_spec = _resolve_need_x_spec(train_cfg, model_cfg)

    loss_type = str(train_cfg.get("loss_type", "ce")).strip().lower()
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    focal_gamma = float(train_cfg.get("focal_gamma", 2.0))
    logit_adjust_tau = float(train_cfg.get("logit_adjust_tau", 0.0))
    class_weight_mode = str(train_cfg.get("class_weight_mode", "none")).strip().lower()
    class_weight_beta = float(train_cfg.get("class_weight_beta", 0.999))

    flat_gt = gt.reshape(-1).astype(np.int64)
    tr_labels = flat_gt[tr_indices] - int(label_offset)
    valid = (tr_labels >= 0) & (tr_labels < num_classes)
    counts = np.bincount(tr_labels[valid], minlength=num_classes).astype(np.float64)
    if float(counts.sum()) <= 0.0:
        counts = np.ones((num_classes,), dtype=np.float64)
    class_prior_np = counts / max(1.0, float(counts.sum()))

    class_weights_np: np.ndarray | None
    if class_weight_mode in ("inv", "inverse"):
        class_weights_np = 1.0 / np.clip(counts, 1.0, None)
    elif class_weight_mode in ("sqrt_inv", "sqrt_inverse"):
        class_weights_np = 1.0 / np.sqrt(np.clip(counts, 1.0, None))
    elif class_weight_mode in ("effective", "effective_num"):
        beta = min(max(class_weight_beta, 0.0), 0.999999)
        class_weights_np = (1.0 - beta) / (1.0 - np.power(beta, np.clip(counts, 1.0, None)))
    else:
        class_weights_np = None
    if class_weights_np is not None:
        class_weights_np = class_weights_np / np.maximum(class_weights_np.mean(), 1.0e-12)

    ds_train_kwargs = dict(
        cube=cube, gt=gt, indices=tr_indices, patch_size=patch_size, mean=mean, std=std,
        label_offset=label_offset,
        augment=bool(train_cfg.get("augment", False)),
        return_x_spec=need_x_spec,
        noise_std=float(train_cfg.get("noise_std", 0.0)),
    )
    ds_eval_kwargs = dict(
        cube=cube, gt=gt, patch_size=patch_size, mean=mean, std=std, label_offset=label_offset, augment=False,
        return_x_spec=need_x_spec,
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
        ds_tr, batch_size=batch_size, shuffle=True,
        drop_last=drop_last, **dl_tr_common
    )
    dl_va = DataLoader(ds_va, batch_size=eval_batch_size, shuffle=False, drop_last=False, **dl_eval_common)
    dl_te = DataLoader(ds_te, batch_size=eval_batch_size, shuffle=False, drop_last=False, **dl_eval_common)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = _maybe_scaler(use_amp)
    model_raw_bands = int(raw_bands)
    model = _build_model(
        model_cfg,
        num_classes=num_classes,
        raw_bands=model_raw_bands,
        device=device,
        patch_size=patch_size,
    )
    init_ckpt_path = Path(str(args.init_ckpt)).resolve() if str(args.init_ckpt).strip() else None
    if init_ckpt_path is not None:
        ck_init = _torch_load_compat(init_ckpt_path, map_location="cpu")
        init_state = ck_init.get(str(args.init_ckpt_key), ck_init)
        if not isinstance(init_state, dict):
            raise ValueError(
                f"--init_ckpt loaded object with unsupported type: {type(init_state)}; "
                f"expected state_dict or dict containing key '{args.init_ckpt_key}'."
            )
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        print(
            f"[init] ckpt={init_ckpt_path} key={args.init_ckpt_key} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            print(f"[init] missing_keys(sample): {missing[:8]}")
        if unexpected:
            print(f"[init] unexpected_keys(sample): {unexpected[:8]}")
    class_prior_t = torch.from_numpy(class_prior_np.astype(np.float32)).to(device)
    class_weights_t = (
        torch.from_numpy(class_weights_np.astype(np.float32)).to(device)
        if class_weights_np is not None
        else None
    )
    num_params_total = int(sum(p.numel() for p in model.parameters()))
    num_params_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(
        f"[run] mode=train model=CenterTargetMamba dataset={ds_name} seed={int(args.seed)} "
        f"device={device.type} amp={use_amp} patch={patch_size} "
        f"split={len(ds_tr)}/{len(ds_va)}/{len(ds_te)}(tr/val/te)"
    )
    print(
        f"[run] workers={num_workers}/{eval_num_workers}(tr/ev) "
        f"deterministic={deterministic} need_x_spec={need_x_spec}"
    )
    print(
        f"[run] loss_type={loss_type} label_smoothing={label_smoothing:.4f} "
        f"focal_gamma={focal_gamma:.3f} logit_adjust_tau={logit_adjust_tau:.3f} "
        f"class_weight_mode={class_weight_mode}"
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

    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    mixup_prob = float(train_cfg.get("mixup_prob", 0.0))

    aug_noise_std = float(train_cfg.get("aug_noise_std", 0.0))

    select_metric = str(train_cfg.get("select_metric", "kappa")).lower()
    final_eval_amp = bool(train_cfg.get("final_eval_amp", True)) and use_amp
    final_eval_num_workers = max(0, int(train_cfg.get("final_eval_num_workers", eval_num_workers)))
    final_eval_prefetch_factor = int(train_cfg.get("final_eval_prefetch_factor", eval_prefetch_factor))
    final_eval_log_interval = max(0, int(train_cfg.get("final_eval_log_interval", 25)))
    final_eval_batch_size = max(1, int(train_cfg.get("final_eval_batch_size", 1024)))

    best_ep = -1
    best_score = -1.0e18

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
            mixup_alpha=mixup_alpha, mixup_prob=mixup_prob,
            aug_noise_std=aug_noise_std,
            patch_size=patch_size,
            need_x_spec=need_x_spec,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            logit_adjust_tau=logit_adjust_tau,
            class_prior=class_prior_t,
            class_weights=class_weights_t,
        )
        loss_out = train_one_epoch(**_filter_kwargs(train_one_epoch, train_kwargs))
        loss = _unwrap_scalar(loss_out)

        sched.step(ep)
        lr_now = float(optimizer.param_groups[0]["lr"])

        dt_ep = time.time() - ep_t0
        dt_total = time.time() - t0
        do_eval = True
        if do_eval:
            eval_kwargs = dict(
                model=model,
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

            improved = (best_ep < 0) or (float(score) > float(best_score))
            if improved:
                best_score = float(score)
                best_ep = ep
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "opt": optimizer.state_dict(),
                    "sched": sched.state_dict(),
                    "epoch": int(ep),
                    "best_score": float(best_score),
                    "meta": {
                        "dataset": ds_name,
                        "split_json": str(Path(args.split_json)),
                        "label_offset": int(label_offset),
                        "num_classes": int(num_classes),
                        "patch_size": int(patch_size),
                        "norm_path": str(out_dir / "meta" / "norm_stats.npz"),
                        "spectral_preprocess_path": str(spectral_preprocess_path),
                    },
                }
                torch.save(ckpt, out_dir / "checkpoints" / "best.pt")

            print(
                f"[train][ep {ep + 1:03d}/{max_epochs:03d}] "
                f"loss={_fmt(loss, '0.6f')} lr={lr_now:.2e} | "
                f"val_OA={_fmt(oa, '0.4f')} val_AA={_fmt(aa, '0.4f')} val_Kappa={_fmt(kp, '0.4f')} | "
                f"score={_fmt(score, '0.4f')} "
                f"best={_fmt(best_score, '0.4f')}@{(best_ep + 1) if best_ep >= 0 else 0:03d} | "
                f"t_ep={dt_ep:.2f}s t_total={dt_total:.1f}s"
            )

    print("[final] loading best checkpoint...")
    ckpt = _torch_load_compat(out_dir / "checkpoints" / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print("[final] ckpt_key=model")

    dl_final_common = dict(
        num_workers=final_eval_num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    if final_eval_num_workers > 0 and final_eval_prefetch_factor > 0:
        dl_final_common["prefetch_factor"] = int(final_eval_prefetch_factor)

    dl_va_final = DataLoader(ds_va, batch_size=final_eval_batch_size, shuffle=False, drop_last=False, **dl_final_common)
    dl_te_final = DataLoader(ds_te, batch_size=final_eval_batch_size, shuffle=False, drop_last=False, **dl_final_common)

    eval_val_kwargs = dict(
        model=model,
        dl=dl_va_final,
        device=device,
        num_classes=num_classes,
        use_amp=final_eval_amp,
        patch_size=patch_size,
        need_x_spec=need_x_spec,
        log_prefix="final_val",
        log_interval=final_eval_log_interval,
    )
    eval_test_kwargs = dict(
        model=model,
        dl=dl_te_final,
        device=device,
        num_classes=num_classes,
        use_amp=final_eval_amp,
        patch_size=patch_size,
        need_x_spec=need_x_spec,
        log_prefix="final_test",
        log_interval=final_eval_log_interval,
    )

    print(
        f"[final][VAL] samples={len(ds_va)} batch={final_eval_batch_size} workers={final_eval_num_workers} amp={final_eval_amp}"
    )
    val_metrics = evaluate(**_filter_kwargs(evaluate, eval_val_kwargs))
    print(
        f"[final][TEST] samples={len(ds_te)} batch={final_eval_batch_size} workers={final_eval_num_workers} amp={final_eval_amp}"
    )
    test_metrics = evaluate(**_filter_kwargs(evaluate, eval_test_kwargs))

    time_sec = float(time.time() - t0)

    final_out = {
        "best_ep": int(best_ep + 1),
        "VAL": val_metrics,
        "TEST": test_metrics,
        "time_sec": time_sec,
        "meta": {
            "dataset": ds_name,
            "split_json": str(split_json_path),
            "seed": int(args.seed),
            "label_offset": int(label_offset),
            "num_classes": int(num_classes),
            "patch_size": int(patch_size),
            "dataset_cfg": str(dataset_cfg_path),
            "model_cfg": str(model_cfg_path),
            "train_cfg": str(train_cfg_path),
            "dataset_cfg_sha1": dataset_cfg_sha1,
            "model_cfg_sha1": model_cfg_sha1,
            "train_cfg_sha1": train_cfg_sha1,
            "split_json_sha1": split_json_sha1,
            "norm_path": str(out_dir / "meta" / "norm_stats.npz"),
            "spectral_preprocess_path": str(spectral_preprocess_path) if spectral_preprocess_path.exists() else "",
            "spectral_preprocess_mode": str(spectral_state.get("mode", "none")),
            "spectral_raw_bands": int(spectral_state.get("raw_bands", raw_bands)),
            "spectral_out_bands": int(spectral_state.get("out_bands", raw_bands)),
            "tta": False,
            "num_params_total": int(num_params_total),
            "num_params_trainable": int(num_params_trainable),
        },
    }
    _save_json(out_dir / "metrics.json", final_out)

    print(
        f"[done] out_dir={out_dir} best_ep={best_ep + 1} ckpt_key=model "
        f"val_OA={_fmt(val_metrics.get('OA', float('nan')), '0.6f')} "
        f"val_AA={_fmt(val_metrics.get('AA', float('nan')), '0.6f')} "
        f"val_Kappa={_fmt(val_metrics.get('Kappa', float('nan')), '0.6f')} "
        f"test_OA={_fmt(test_metrics.get('OA', float('nan')), '0.6f')} "
        f"test_AA={_fmt(test_metrics.get('AA', float('nan')), '0.6f')} "
        f"test_Kappa={_fmt(test_metrics.get('Kappa', float('nan')), '0.6f')} "
        f"time_sec={time_sec:.2f}"
    )


if __name__ == "__main__":
    main()
