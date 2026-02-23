#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _as_x_xspec_y(batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Accept:
      (x, y)
      (x, x_spec, y, ...)
      (x, y, x_spec, ...)
      {"x":..., "y":..., "x_spec":...} (or "spec")
    """
    if isinstance(batch, dict):
        x = batch.get("x", None)
        y = batch.get("y", None)
        x_spec = batch.get("x_spec", batch.get("spec", None))
        if x is None or y is None:
            vals = [v for v in batch.values() if torch.is_tensor(v)]
            if len(vals) < 2:
                raise ValueError("Batch dict must contain x/y or at least two tensors.")
            # heuristic: y is integer tensor
            y = next((v for v in vals if v.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8)), None)
            if y is None:
                x, y = vals[0], vals[1]
            else:
                x = next(v for v in vals if v is not y)
        return x, x_spec, y

    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch tuple must have at least (x, y).")
        x = batch[0]
        x_spec = None
        y = None

        # common patterns
        if len(batch) >= 3:
            b1, b2 = batch[1], batch[2]
            if torch.is_tensor(b1) and b1.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
                y = b1
                if torch.is_tensor(b2) and b2.dtype.is_floating_point:
                    x_spec = b2
            elif torch.is_tensor(b2) and b2.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
                y = b2
                if torch.is_tensor(b1) and b1.dtype.is_floating_point:
                    x_spec = b1

        if y is None:
            y = batch[1]

        return x, x_spec, y

    raise ValueError(f"Unsupported batch type: {type(batch)}")


def _find_patch_spatial_dims(x: torch.Tensor, patch_size: int = 15) -> Tuple[int, int]:
    dims = [i for i in range(1, x.ndim) if x.shape[i] == patch_size]
    if len(dims) >= 2:
        return dims[-2], dims[-1]
    return x.ndim - 2, x.ndim - 1


def _find_spectral_dim(x: torch.Tensor, patch_size: int = 15) -> int:
    spatial = set(_find_patch_spatial_dims(x, patch_size))
    cand = []
    for i in range(1, x.ndim):
        if i in spatial:
            continue
        cand.append((int(x.shape[i]), i))
    cand.sort(reverse=True)
    return cand[0][1] if cand else 1


def _derive_x_spec_from_x(x: torch.Tensor, patch_size: int = 15) -> Optional[torch.Tensor]:
    if not torch.is_tensor(x):
        return None
    if x.ndim < 3:
        return None

    # Fast-path: (B,C,H,W) or (B,H,W,C)
    if x.ndim == 4:
        if int(x.shape[2]) == patch_size and int(x.shape[3]) == patch_size:
            return x[:, :, patch_size // 2, patch_size // 2]
        if int(x.shape[1]) == patch_size and int(x.shape[2]) == patch_size:
            return x[:, patch_size // 2, patch_size // 2, :]

    hdim, wdim = _find_patch_spatial_dims(x, patch_size)
    sd = _find_spectral_dim(x, patch_size)

    h = int(x.shape[hdim])
    w = int(x.shape[wdim])
    ch, cw = h // 2, w // 2

    slc = [slice(None)] * x.ndim
    slc[hdim] = ch
    slc[wdim] = cw
    xc = x[tuple(slc)]

    # xc should be (B, bands) in most cases; otherwise, try to squeeze/reshape.
    if xc.ndim == 2:
        return xc
    if xc.ndim > 2:
        # put spectral dim to dim=1 if possible
        # after selecting spatial dims, original spectral dim shifts if it was after them
        # fall back: take the largest non-batch dim as bands
        if xc.ndim >= 3:
            sizes = [(int(xc.shape[i]), i) for i in range(1, xc.ndim)]
            sizes.sort(reverse=True)
            bands_dim = sizes[0][1]
            xc = xc.movedim(bands_dim, 1)
            xc = xc.reshape(xc.shape[0], xc.shape[1], -1).mean(dim=-1)
            return xc
    return None


def _apply_spec_dropout(x: torch.Tensor, p: float, ratio: float, patch_size: int = 15) -> torch.Tensor:
    if p <= 0.0 or ratio <= 0.0:
        return x
    if torch.rand((), device=x.device).item() > p:
        return x
    sd = _find_spectral_dim(x, patch_size)
    n_bands = int(x.shape[sd])
    if n_bands <= 1:
        return x
    width = max(1, int(round(n_bands * ratio)))
    start = int(torch.randint(0, max(1, n_bands - width + 1), (), device=x.device).item())
    slc = [slice(None)] * x.ndim
    slc[sd] = slice(start, start + width)
    out = x.clone()
    out[tuple(slc)] = 0.0
    return out


def _mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    x_spec: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0:
        return x, x_spec, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = float(max(0.0, min(1.0, lam)))
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y2 = y[idx]
    x_mix = x * lam + x2 * (1.0 - lam)
    x_spec_mix = None
    if x_spec is not None:
        x_spec2 = x_spec[idx]
        x_spec_mix = x_spec * lam + x_spec2 * (1.0 - lam)
    return x_mix, x_spec_mix, y, y2, lam


def _ce_loss(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    return F.cross_entropy(logits, y, label_smoothing=float(label_smoothing))


def _focal_loss(logits: torch.Tensor, y: torch.Tensor, gamma: float, label_smoothing: float = 0.0) -> torch.Tensor:
    ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=float(label_smoothing))
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** float(gamma)) * ce
    return loss.mean()


def _forward_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    x_spec: Optional[torch.Tensor],
    *,
    patch_size: int = 15,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # derive x_spec if missing (for models that need it)
    if x_spec is None:
        x_spec = _derive_x_spec_from_x(x, patch_size=patch_size)

    out = None
    # try (x, x_spec) first if we have x_spec; otherwise try (x)
    if x_spec is not None:
        try:
            out = model(x, x_spec)
        except TypeError:
            out = model(x)
    else:
        try:
            out = model(x)
        except TypeError:
            # last resort: derive and try again
            x_spec2 = _derive_x_spec_from_x(x, patch_size=patch_size)
            if x_spec2 is None:
                raise
            out = model(x, x_spec2)

    unc = None
    if torch.is_tensor(out):
        return out, None
    if isinstance(out, (list, tuple)):
        logits = out[0]
        for t in out[1:]:
            if torch.is_tensor(t) and t.ndim == 1:
                unc = t
                break
        return logits, unc
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
        for k in ("uncertainty", "uncert", "u"):
            if k in out and torch.is_tensor(out[k]):
                unc = out[k].detach()
                break
        return logits, unc
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
    label_smoothing: float = 0.0,
    conf_penalty: float = 0.0,
    focal_gamma: float = 0.0,
    mixup_prob: float = 0.0,
    mixup_alpha: float = 0.0,
    spec_dropout_p: float = 0.0,
    spec_dropout_ratio: float = 0.0,
    aug_noise_std: float = 0.0,
    patch_size: int = 15,
    grad_accum_steps: int = 1,
    steps_per_epoch: int = 0,
    epoch: int = 0,
    **kwargs,
) -> float:
    model.train()
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    grad_accum_steps = max(1, int(grad_accum_steps))
    losses = []
    optimizer.zero_grad(set_to_none=True)

    it_max = None
    if steps_per_epoch and int(steps_per_epoch) > 0:
        it_max = int(steps_per_epoch)

    for it, batch in enumerate(dl):
        if it_max is not None and it >= it_max:
            break

        x, x_spec, y = _as_x_xspec_y(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        if x_spec is not None and torch.is_tensor(x_spec):
            x_spec = x_spec.to(device, non_blocking=True)

        if aug_noise_std and float(aug_noise_std) > 0.0:
            x = x + torch.randn_like(x) * float(aug_noise_std)
            if x_spec is not None:
                x_spec = x_spec + torch.randn_like(x_spec) * float(aug_noise_std)

        x = _apply_spec_dropout(x, float(spec_dropout_p), float(spec_dropout_ratio), patch_size=patch_size)
        if x_spec is None:
            x_spec = _derive_x_spec_from_x(x, patch_size=patch_size)
        else:
            # keep spec in sync if dropout was applied on x
            x_spec = x_spec

        do_mix = (mixup_prob > 0.0) and (mixup_alpha > 0.0) and (torch.rand((), device=device).item() < mixup_prob)
        if do_mix:
            x, x_spec, y_a, y_b, lam = _mixup(x, y, float(mixup_alpha), x_spec=x_spec)
        else:
            y_a, y_b, lam = y, y, 1.0

        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits, _ = _forward_logits(model, x, x_spec, patch_size=patch_size)

            if focal_gamma and float(focal_gamma) > 0.0:
                loss_a = _focal_loss(logits, y_a, float(focal_gamma), label_smoothing=label_smoothing)
                if lam < 1.0:
                    loss_b = _focal_loss(logits, y_b, float(focal_gamma), label_smoothing=label_smoothing)
                    loss = lam * loss_a + (1.0 - lam) * loss_b
                else:
                    loss = loss_a
            else:
                loss_a = _ce_loss(logits, y_a, label_smoothing=label_smoothing)
                if lam < 1.0:
                    loss_b = _ce_loss(logits, y_b, label_smoothing=label_smoothing)
                    loss = lam * loss_a + (1.0 - lam) * loss_b
                else:
                    loss = loss_a

            if conf_penalty and float(conf_penalty) > 0.0:
                p = torch.softmax(logits, dim=1)
                neg_entropy = (p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
                loss = loss + float(conf_penalty) * neg_entropy

            loss = loss / float(grad_accum_steps)

        scaler.scale(loss).backward()

        if (it + 1) % grad_accum_steps == 0:
            if grad_clip and float(grad_clip) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().item() * float(grad_accum_steps)))

    # flush remainder
    if losses and ((it + 1) % grad_accum_steps != 0):
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


def evaluate(
    model: torch.nn.Module,
    dl: Iterable,
    device: torch.device,
    *,
    num_classes: int,
    use_amp: bool = False,
    patch_size: int = 15,
    steps: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    unc_list = []

    it_max = int(steps) if steps and int(steps) > 0 else None

    with torch.no_grad():
        for it, batch in enumerate(dl):
            if it_max is not None and it >= it_max:
                break
            x, x_spec, y = _as_x_xspec_y(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            if x_spec is not None and torch.is_tensor(x_spec):
                x_spec = x_spec.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits, unc = _forward_logits(model, x, x_spec, patch_size=patch_size)

            pred = torch.argmax(logits, dim=1)
            _confusion_matrix_update(cm, y.detach().cpu().numpy(), pred.detach().cpu().numpy(), num_classes)

            if unc is not None and torch.is_tensor(unc):
                unc_list.append(unc.detach().float().cpu().numpy().reshape(-1))

    out = _metrics_from_cm(cm)
    if unc_list:
        u = np.concatenate(unc_list, axis=0)
        out["uncertainty_mean"] = float(u.mean())
        out["uncertainty_std"] = float(u.std())
    return out
