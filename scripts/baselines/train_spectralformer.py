#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(8):
        if (p / "configs").exists() and (p / "src").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]


repo_root = _repo_root()
sys.path.insert(0, str(repo_root))

from hsi3d.utils.io import load_yaml, load_json, ensure_dir, save_json  # noqa: E402
from hsi3d.utils.seed import set_global_seed  # noqa: E402
from hsi3d.data.hsi_dataset import compute_train_norm  # noqa: E402
from hsi3d.metrics.classification import classification_report  # noqa: E402


def _short_name(dataset: str) -> str:
    return {"pavia_university": "pu"}.get(dataset, dataset)


def _sha1_of_dict(d: Dict[str, Any]) -> str:
    b = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


def _load_vit(sf_dir: Path):
    vit_path = sf_dir / "vit_pytorch.py"
    if not vit_path.exists():
        raise FileNotFoundError(f"[spectralformer] missing {vit_path}")

    spec = importlib.util.spec_from_file_location("spectralformer_vit_pytorch", str(vit_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"[spectralformer] failed to create module spec for {vit_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, "ViT"):
        raise AttributeError(f"[spectralformer] ViT not found in {vit_path}")

    return mod.ViT


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(key)
    return d[key]


def _require_keys(d: Dict[str, Any], keys: Tuple[str, ...], ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{ctx} missing keys: {missing}")


def _make_band_mat(B: int, near_band: int) -> np.ndarray:
    nb = int(near_band)
    if nb <= 0:
        raise ValueError(f"near_band must be > 0 (got {near_band})")
    half = nb // 2
    mat = np.zeros((B, nb), dtype=np.int64)
    for i in range(B):
        start = i - half
        for j in range(nb):
            k = start + j
            if k < 0:
                k = 0
            elif k >= B:
                k = B - 1
            mat[i, j] = k
    return mat


class SpectralFormerTokenDataset(Dataset):
    def __init__(
        self,
        cube_pad: np.ndarray,
        gt: np.ndarray,
        indices_flat: np.ndarray,
        *,
        H: int,
        W: int,
        B: int,
        patch_size: int,
        label_offset: int,
        band_mat: np.ndarray,
    ) -> None:
        super().__init__()
        self.cube_pad = cube_pad
        self.gt = gt
        self.indices = np.asarray(indices_flat, dtype=np.int64).reshape(-1)
        self.H = int(H)
        self.W = int(W)
        self.B = int(B)
        self.ps = int(patch_size)
        self.label_offset = int(label_offset)
        self.band_mat = np.asarray(band_mat, dtype=np.int64)
        self.patch_dim = self.ps * self.ps * self.band_mat.shape[1]

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        r = idx // self.W
        c = idx - r * self.W

        patch = self.cube_pad[r : r + self.ps, c : c + self.ps, :]  # (ps, ps, B)
        arr = patch[:, :, self.band_mat]  # (ps, ps, B, near_band)
        tok = np.transpose(arr, (2, 0, 1, 3)).reshape(self.B, self.patch_dim).astype(np.float32, copy=False)

        y = int(self.gt[r, c]) - self.label_offset
        return torch.from_numpy(tok), torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def _eval_model(model: nn.Module, dl: DataLoader, num_classes: int, device: torch.device) -> Dict[str, Any]:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        yt = y.detach().cpu().numpy()
        for a, b in zip(yt, pred):
            if 0 <= a < num_classes and 0 <= b < num_classes:
                cm[a, b] += 1
    return classification_report(cm)


def _score(m: Dict[str, Any], select_metric: str) -> float:
    k = select_metric.strip().lower()
    if k == "oa":
        return float(m["OA"])
    if k == "aa":
        return float(m["AA"])
    return float(m["Kappa"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)

    # optional consistency checks (NOT used to set hyperparams)
    ap.add_argument("--patch_size", type=int, default=-1)
    ap.add_argument("--use_amp", action="store_true")
    args = ap.parse_args()

    set_global_seed(int(args.seed))

    dataset_cfg_path = Path(args.dataset_cfg)
    dataset = load_yaml(str(dataset_cfg_path))

    dataset_name = str(dataset.get("name") or dataset.get("dataset") or "")
    if not dataset_name:
        raise KeyError("[dataset_cfg] missing 'name' or 'dataset'")

    processed_dir = dataset.get("processed_dir", None)
    if processed_dir is None:
        raise KeyError("[dataset_cfg] missing 'processed_dir'")

    processed_dir = Path(str(processed_dir))
    cube_path = repo_root / processed_dir / "cube.npy"
    gt_path = repo_root / processed_dir / "gt.npy"
    if not cube_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"[spectralformer] missing processed npy: {cube_path} / {gt_path}")

    label_offset = int(dataset.get("label_offset", 1))
    num_classes = int(dataset.get("num_classes", 0))
    if num_classes <= 1:
        raise ValueError(f"[spectralformer] invalid num_classes={num_classes}")

    # ---- STRICT baseline config: configs/baselines/<short>.yaml ----
    bcfg_path = repo_root / "configs" / "baselines" / f"{_short_name(dataset_name)}.yaml"
    if not bcfg_path.exists():
        raise FileNotFoundError(f"[spectralformer] missing baseline cfg: {bcfg_path}")

    bcfg_all = load_yaml(str(bcfg_path))
    baseline_root = bcfg_all.get("baseline", None)
    if baseline_root is None:
        raise KeyError(f"[baseline cfg] missing top key: baseline ({bcfg_path})")

    sf = baseline_root.get("spectralformer", None)
    if sf is None:
        raise KeyError(f"[baseline cfg] missing key: baseline.spectralformer ({bcfg_path})")

    REQUIRED = (
        "patch_size",
        "near_band",
        "mode",
        "dim",
        "depth",
        "heads",
        "dim_head",
        "mlp_dim",
        "dropout",
        "emb_dropout",
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "lr",
        "weight_decay",
        "betas",
        "label_smoothing",
        "grad_clip",
        "max_epochs",
        "early_stop",
        "early_stop_patience",
        "min_epochs",
        "select_metric",
        "scheduler",
        "min_lr",
        "pad_mode",
        "norm_mean_global_blend",
        "norm_std_global_ratio",
        "norm_std_abs_floor",
        "use_amp",
    )
    _require_keys(sf, REQUIRED, ctx=f"[baseline cfg] baseline.spectralformer ({bcfg_path})")

    ps = int(sf["patch_size"])
    if args.patch_size != -1 and int(args.patch_size) != ps:
        raise ValueError(f"[spectralformer] CLI patch_size={args.patch_size} != yaml patch_size={ps}")

    use_amp_yaml = bool(sf["use_amp"])
    if args.use_amp and not use_amp_yaml:
        raise ValueError("[spectralformer] CLI --use_amp is set but yaml baseline.spectralformer.use_amp=false")

    near_band = int(sf["near_band"])
    mode = str(sf["mode"])

    dim = int(sf["dim"])
    depth = int(sf["depth"])
    heads = int(sf["heads"])
    dim_head = int(sf["dim_head"])
    mlp_dim = int(sf["mlp_dim"])
    dropout = float(sf["dropout"])
    emb_dropout = float(sf["emb_dropout"])

    batch_size = int(sf["batch_size"])
    eval_batch_size = int(sf["eval_batch_size"])
    num_workers = int(sf["num_workers"])

    lr = float(sf["lr"])
    weight_decay = float(sf["weight_decay"])
    betas = sf["betas"]
    if not (isinstance(betas, (list, tuple)) and len(betas) == 2):
        raise ValueError("[spectralformer] yaml betas must be a list/tuple of length 2")
    beta1 = float(betas[0])
    beta2 = float(betas[1])

    label_smoothing = float(sf["label_smoothing"])
    grad_clip = float(sf["grad_clip"])

    max_epochs = int(sf["max_epochs"])
    early_stop = bool(sf["early_stop"])
    early_stop_patience = int(sf["early_stop_patience"])
    min_epochs = int(sf["min_epochs"])
    select_metric = str(sf["select_metric"])

    scheduler_name = str(sf["scheduler"]).strip().lower()
    min_lr = float(sf["min_lr"])

    pad_mode = str(sf["pad_mode"])

    norm_mean_global_blend = float(sf["norm_mean_global_blend"])
    norm_std_global_ratio = float(sf["norm_std_global_ratio"])
    norm_std_abs_floor = float(sf["norm_std_abs_floor"])

    # ---- split ----
    split = load_json(args.split_json)
    train_idx = np.asarray(_require(split, "train_indices"), dtype=np.int64)
    val_idx = np.asarray(_require(split, "val_indices"), dtype=np.int64)
    test_idx = np.asarray(_require(split, "test_indices"), dtype=np.int64)

    # ---- data ----
    cube = np.load(cube_path)
    gt = np.load(gt_path)
    if cube.ndim != 3 or gt.ndim != 2:
        raise ValueError(f"[spectralformer] bad shapes cube={cube.shape}, gt={gt.shape}")

    H, W = int(gt.shape[0]), int(gt.shape[1])
    if cube.shape[0] != H or cube.shape[1] != W:
        raise ValueError(f"[spectralformer] cube spatial mismatch: cube={cube.shape} gt={gt.shape}")

    B = int(cube.shape[2])

    mean, std = compute_train_norm(
        cube,
        train_idx,
        mean_global_blend=norm_mean_global_blend,
        std_global_ratio=norm_std_global_ratio,
        std_abs_floor=norm_std_abs_floor,
    )
    cube = (cube.astype(np.float32, copy=False) - mean.reshape(1, 1, B)) / std.reshape(1, 1, B)

    pad = ps // 2
    cube_pad = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode=pad_mode)
    band_mat = _make_band_mat(B, near_band)

    ds_train = SpectralFormerTokenDataset(cube_pad, gt, train_idx, H=H, W=W, B=B, patch_size=ps, label_offset=label_offset, band_mat=band_mat)
    ds_val = SpectralFormerTokenDataset(cube_pad, gt, val_idx, H=H, W=W, B=B, patch_size=ps, label_offset=label_offset, band_mat=band_mat)
    ds_test = SpectralFormerTokenDataset(cube_pad, gt, test_idx, H=H, W=W, B=B, patch_size=ps, label_offset=label_offset, band_mat=band_mat)

    if len(ds_train) == 0 or len(ds_val) == 0 or len(ds_test) == 0:
        raise ValueError(f"[spectralformer] empty split: train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    # ---- model ----
    sf_dir = repo_root / "external" / "baselines" / "spectralformer"
    ViT = _load_vit(sf_dir)

    sig = inspect.signature(ViT.__init__)
    kwargs: Dict[str, Any] = {}
    for k, v in [
        ("image_size", ps),
        ("near_band", near_band),
        ("num_patches", B),
        ("num_classes", num_classes),
        ("dim", dim),
        ("depth", depth),
        ("heads", heads),
        ("mlp_dim", mlp_dim),
        ("dim_head", dim_head),
        ("dropout", dropout),
        ("emb_dropout", emb_dropout),
        ("mode", mode),
        ("pool", "cls"),
    ]:
        if k in sig.parameters:
            kwargs[k] = v

    model = ViT(**kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=min_lr)
    elif scheduler_name == "step":
        raise ValueError("[spectralformer] scheduler=step is not implemented; please use scheduler=cosine")
    else:
        raise ValueError(f"[spectralformer] unknown scheduler={scheduler_name}")

    # AMP (yaml-controlled)
    use_amp = bool(use_amp_yaml)

    try:
        from torch.amp import GradScaler, autocast  # type: ignore
        scaler = GradScaler("cuda", enabled=use_amp)
        autocast_ctx = lambda: autocast(device_type="cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    best_score = -1e9
    best_ep = -1
    noimp = 0
    best_path = out_dir / "best.pt"

    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(x)
                loss = criterion(logits, y)

            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            running += float(loss.detach().item()) * int(y.numel())
            n_seen += int(y.numel())

        scheduler.step()

        val_m = _eval_model(model, dl_val, num_classes, device)
        sc = _score(val_m, select_metric)

        if sc > best_score + 1e-12:
            best_score = sc
            best_ep = ep
            noimp = 0
            torch.save({"model": model.state_dict()}, best_path)
        else:
            noimp += 1

        if ep == 1 or ep % 10 == 0:
            dt = time.time() - t0
            loss_ep = running / max(1, n_seen)
            print(
                f"[ep {ep:04d}] loss={loss_ep:.6f} | "
                f"VAL OA={val_m['OA']:.4f} AA={val_m['AA']:.4f} Kappa={val_m['Kappa']:.4f} "
                f"score={sc:.4f} best={best_score:.4f}@{best_ep} noimp={noimp} dt={dt:.1f}s"
            )

        if early_stop and ep >= min_epochs and noimp >= early_stop_patience:
            print(f"[early_stop] no improve for {noimp} epochs (best_ep={best_ep})")
            break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)

    val_best = _eval_model(model, dl_val, num_classes, device)
    test_best = _eval_model(model, dl_test, num_classes, device)

    meta = {
        "dataset": dataset_name,
        "baseline": "spectralformer",
        "split_json": str(Path(args.split_json)),
        "ckpt": str(best_path),
        "label_offset": label_offset,
        "num_classes": num_classes,
        "patch_size": ps,
        "near_band": near_band,
        "mode": mode,
        "cfg_path": str(bcfg_path),
        "cfg_sha1": _sha1_of_dict(sf),
        "use_amp": bool(use_amp),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }

    metrics = {"VAL": val_best, "TEST": test_best, "meta": meta}
    save_json(out_dir / "metrics.json", metrics)

    print("[done] out_dir=", out_dir)
    print("[val ]", {k: val_best[k] for k in ("OA", "AA", "Kappa")})
    print("[test]", {k: test_best[k] for k in ("OA", "AA", "Kappa")})


if __name__ == "__main__":
    main()
