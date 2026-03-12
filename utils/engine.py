#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _as_x_y(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize a supported batch layout to `(x, y)`."""
    if isinstance(batch, dict):
        x = batch.get("x", None)
        y = batch.get("y", None)
        if x is None or y is None:
            vals = [v for v in batch.values() if torch.is_tensor(v)]
            if len(vals) < 2:
                raise ValueError("Batch dict must contain x/y or at least two tensors.")
            y = next((v for v in vals if v.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8)), None)
            if y is None:
                x, y = vals[0], vals[1]
            else:
                x = next(v for v in vals if v is not y)
        return x, y

    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch tuple must have at least (x, y).")
        x = batch[0]
        y = None

        if len(batch) >= 3:
            b1, b2 = batch[1], batch[2]
            if torch.is_tensor(b1) and b1.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
                y = b1
            elif torch.is_tensor(b2) and b2.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
                y = b2

        if y is None:
            y = batch[1]

        return x, y

    raise ValueError(f"Unsupported batch type: {type(batch)}")


def _mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = float(max(0.0, min(1.0, lam)))
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y2 = y[idx]
    x_mix = x * lam + x2 * (1.0 - lam)
    return x_mix, y, y2, lam


def _ce_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        y,
        weight=class_weights,
        label_smoothing=float(max(0.0, label_smoothing)),
    )


def _focal_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    gamma = float(max(0.0, gamma))
    log_prob = F.log_softmax(logits, dim=1)
    prob = log_prob.exp()
    y = y.long()
    pt = prob.gather(dim=1, index=y.unsqueeze(1)).squeeze(1).clamp_min(1.0e-8)
    log_pt = torch.log(pt)
    focal_factor = (1.0 - pt).pow(gamma)
    loss = -focal_factor * log_pt
    if class_weights is not None:
        w = class_weights[y]
        loss = loss * w
    return loss.mean()


def _forward_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    out = model(x)

    if torch.is_tensor(out):
        return out, None
    if isinstance(out, (list, tuple)):
        logits = out[0]
        aux = None
        for t in out[1:]:
            if torch.is_tensor(t) and t.ndim == 1:
                aux = t
                break
        return logits, aux
    if isinstance(out, dict):
        for k in ("logits", "pred", "y", "scores"):
            if k in out and torch.is_tensor(out[k]):
                logits = out[k]
                break
        else:
            tvals = [v for v in out.values() if torch.is_tensor(v)]
            if not tvals:
                raise ValueError("Model output dict has no tensors.")
            logits = tvals[0]
        return logits, None
    raise ValueError(f"Unsupported model output type: {type(out)}")


def train_one_epoch(
    model: torch.nn.Module,
    dl: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
    grad_clip: float = 0.0,
    mixup_prob: float = 0.0,
    mixup_alpha: float = 0.0,
    aug_noise_std: float = 0.0,
    grad_accum_steps: int = 1,
    loss_type: str = "ce",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    grad_accum_steps = max(1, int(grad_accum_steps))
    losses = []
    optimizer.zero_grad(set_to_none=True)

    steps_done = 0
    loss_type = str(loss_type).strip().lower()
    label_smoothing = float(max(0.0, label_smoothing))
    focal_gamma = float(max(0.0, focal_gamma))
    if class_weights is not None:
        class_weights = class_weights.to(device=device, dtype=torch.float32)

    for batch in dl:
        x, y = _as_x_y(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        if aug_noise_std and float(aug_noise_std) > 0.0:
            x = x + torch.randn_like(x) * float(aug_noise_std)

        do_mix = (mixup_prob > 0.0) and (mixup_alpha > 0.0) and (torch.rand((), device=device).item() < mixup_prob)
        if do_mix:
            x, y_a, y_b, lam = _mixup(x, y, float(mixup_alpha))
        else:
            y_a, y_b, lam = y, y, 1.0

        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits, _ = _forward_logits(model, x)

            if loss_type == "focal":
                loss_a = _focal_loss(
                    logits,
                    y_a,
                    gamma=focal_gamma,
                    class_weights=class_weights,
                )
            else:
                loss_a = _ce_loss(
                    logits,
                    y_a,
                    class_weights=class_weights,
                    label_smoothing=label_smoothing,
                )
            if lam < 1.0:
                if loss_type == "focal":
                    loss_b = _focal_loss(
                        logits,
                        y_b,
                        gamma=focal_gamma,
                        class_weights=class_weights,
                    )
                else:
                    loss_b = _ce_loss(
                        logits,
                        y_b,
                        class_weights=class_weights,
                        label_smoothing=label_smoothing,
                    )
                loss = lam * loss_a + (1.0 - lam) * loss_b
            else:
                loss = loss_a

            loss = loss / float(grad_accum_steps)

        scaler.scale(loss).backward()

        if (steps_done + 1) % grad_accum_steps == 0:
            if grad_clip and float(grad_clip) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().item() * float(grad_accum_steps)))
        steps_done += 1

    if losses and (steps_done % grad_accum_steps != 0):
        if grad_clip and float(grad_clip) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return float(np.mean(losses)) if losses else 0.0


def _confusion_matrix_update(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> None:
    k = (y_true >= 0) & (y_true < num_classes)
    y_true = y_true[k]
    y_pred = y_pred[k]
    idx = y_true * num_classes + y_pred
    binc = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    cm += binc


def _metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    n = int(cm.sum())
    if n <= 0:
        return {"OA": 0.0, "AA": 0.0, "Kappa": 0.0, "per_class_acc": [], "confusion_matrix": cm.tolist()}

    diag = np.diag(cm).astype(np.float64)
    row = cm.sum(axis=1).astype(np.float64)
    col = cm.sum(axis=0).astype(np.float64)

    oa = float(diag.sum() / n)
    per = np.divide(diag, row, out=np.zeros_like(diag), where=row > 0)
    aa = float(per[row > 0].mean()) if np.any(row > 0) else 0.0

    pe = float((row * col).sum() / (n * n))
    denom = (1.0 - pe)
    kappa = float((oa - pe) / denom) if denom > 1e-12 else 0.0

    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "per_class_acc": per.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def _can_use_fast_dataset_eval(dl: Iterable) -> bool:
    ds = getattr(dl, "dataset", None)
    if ds is None:
        return False
    if bool(getattr(ds, "augment", False)):
        return False
    if getattr(ds, "_cube_pad", None) is None:
        return False
    needed = ("indices", "gt", "w", "half", "patch_size", "label_offset")
    return all(hasattr(ds, k) for k in needed)


def _evaluate_hsi_dataset_fast(
    model: torch.nn.Module,
    dl: Iterable,
    device: torch.device,
    *,
    num_classes: int,
    use_amp: bool = False,
    patch_size: int = 15,
    steps: int = 0,
    log_prefix: str = "",
    log_interval: int = 0,
) -> Dict[str, Any]:
    ds = getattr(dl, "dataset")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    indices = np.asarray(ds.indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return _metrics_from_cm(cm)

    gt = np.asarray(ds.gt, dtype=np.int64)
    cube_pad = np.asarray(ds._cube_pad, dtype=np.float32)
    w = int(ds.w)
    ps = int(ds.patch_size) if int(ds.patch_size) > 0 else int(patch_size)
    label_offset = int(ds.label_offset)

    bs = int(getattr(dl, "batch_size", 0) or 0)
    if bs <= 0:
        bs = 512

    total_steps = (int(indices.size) + bs - 1) // bs
    it_max = int(steps) if steps and int(steps) > 0 else None
    if it_max is not None:
        total_steps = min(total_steps, max(0, int(it_max)))

    lp = str(log_prefix).strip()
    li = max(0, int(log_interval))
    offsets = np.arange(ps, dtype=np.int64)

    pred_all = np.empty((int(indices.size),), dtype=np.int64)
    r_all = indices // w
    c_all = indices % w

    with torch.no_grad():
        for it in range(total_steps):
            start = it * bs
            end = min(int(indices.size), start + bs)
            r = r_all[start:end]
            c = c_all[start:end]

            rr = r[:, None] + offsets[None, :]
            cc = c[:, None] + offsets[None, :]
            patches = cube_pad[rr[:, :, None], cc[:, None, :], :]
            x_np = np.ascontiguousarray(np.transpose(patches, (0, 3, 1, 2)), dtype=np.float32)

            x_cpu = torch.from_numpy(x_np)
            if device.type == "cuda":
                x_cpu = x_cpu.pin_memory()
            x = x_cpu.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits, _ = _forward_logits(model, x)

            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64, copy=False)
            pred_all[start:end] = pred

            if li > 0 and (((it + 1) % li == 0) or (it == 0)):
                done = it + 1
                p = f"{done / max(1, total_steps):.1%}"
                tag = f"[{lp}] " if lp else ""
                print(f"{tag}eval {done}/{total_steps} ({p})", flush=True)

    y_np = gt[r_all, c_all].astype(np.int64, copy=False) - label_offset
    _confusion_matrix_update(cm, y_np, pred_all, num_classes)
    return _metrics_from_cm(cm)


def evaluate(
    model: torch.nn.Module,
    dl: Iterable,
    device: torch.device,
    *,
    num_classes: int,
    use_amp: bool = False,
    patch_size: int = 15,
    steps: int = 0,
    log_prefix: str = "",
    log_interval: int = 0,
) -> Dict[str, Any]:
    model.eval()
    if _can_use_fast_dataset_eval(dl):
        try:
            return _evaluate_hsi_dataset_fast(
                model=model,
                dl=dl,
                device=device,
                num_classes=num_classes,
                use_amp=use_amp,
                patch_size=patch_size,
                steps=steps,
                log_prefix=log_prefix,
                log_interval=log_interval,
            )
        except Exception as e:
            tag = f"[{str(log_prefix).strip()}] " if str(log_prefix).strip() else ""
            print(f"{tag}[warn] fast_eval fallback to dataloader path: {e}", flush=True)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    it_max = int(steps) if steps and int(steps) > 0 else None
    total_steps = None
    try:
        total_steps = len(dl)
    except Exception:
        total_steps = None

    lp = str(log_prefix).strip()
    li = max(0, int(log_interval))

    with torch.no_grad():
        for it, batch in enumerate(dl):
            if it_max is not None and it >= it_max:
                break
            x, y = _as_x_y(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits, _ = _forward_logits(model, x)

            pred = torch.argmax(logits, dim=1)
            _confusion_matrix_update(cm, y.detach().cpu().numpy(), pred.detach().cpu().numpy(), num_classes)

            if li > 0 and (((it + 1) % li == 0) or (it == 0)):
                done = it + 1
                if total_steps is None:
                    p = "?"
                else:
                    p = f"{done / max(1, total_steps):.1%}"
                tag = f"[{lp}] " if lp else ""
                denom = str(total_steps) if total_steps is not None else "?"
                print(f"{tag}eval {done}/{denom} ({p})", flush=True)

    out = _metrics_from_cm(cm)
    return out
