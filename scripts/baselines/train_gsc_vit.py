#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, Tuple

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

for extra in (repo_root, repo_root / "src"):
    extra_s = str(extra)
    if extra_s not in sys.path:
        sys.path.insert(0, extra_s)

from hsi3d.data.hsi_dataset import HSIPatchDataset, compute_train_norm  # noqa: E402
from hsi3d.utils.io import ensure_dir, load_json, load_yaml, save_json  # noqa: E402
from hsi3d.utils.seed import set_global_seed  # noqa: E402


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_disjoint(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    inter = np.intersect1d(a, b, assume_unique=False)
    if inter.size:
        raise ValueError(f"split overlap: {name_a} ∩ {name_b} size={int(inter.size)}")


def _cm_update(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> None:
    k = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    binc = np.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    cm += binc


def _cm_scores(cm: np.ndarray) -> Tuple[float, float, float]:
    cm = cm.astype(np.float64)
    total = cm.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    po = float(np.trace(cm) / total)
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = float((row * col).sum() / (total * total + 1e-12))
    kappa = float((po - pe) / (1.0 - pe + 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_i = np.diag(cm) / (row + 1e-12)
    aa = float(np.nanmean(acc_i))
    oa = float(po)
    return oa, aa, kappa


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module | None = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    total_loss = 0.0
    n_seen = 0

    for x, _x_spec, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)

        pred = logits.argmax(dim=1).detach().cpu().numpy()
        yt = y.detach().cpu().numpy()
        _cm_update(cm, yt, pred, num_classes)

        if criterion is not None:
            loss = criterion(logits.float(), y)
            bs = int(x.shape[0])
            total_loss += float(loss.detach().item()) * bs
            n_seen += bs

    oa, aa, kappa = _cm_scores(cm)
    out: Dict[str, float] = {"OA": oa, "AA": aa, "Kappa": kappa}
    if criterion is not None and n_seen > 0:
        out["loss"] = float(total_loss / n_seen)
    return out


def _get(d: Dict[str, Any], keys: Tuple[str, ...], default: Any) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()


def _load_mat(path: Path, key: str, want: str) -> np.ndarray:
    key = str(key or "").strip()

    try:
        import scipy.io  # type: ignore
    except Exception:
        scipy = None  # type: ignore
    else:
        scipy = scipy.io  # type: ignore

    if scipy is not None:
        d = scipy.loadmat(str(path))
        d = {k: v for k, v in d.items() if not k.startswith("__")}
        if key:
            if key not in d:
                raise KeyError(f"key '{key}' not found in {path.name}")
            return np.asarray(d[key])
        cand = []
        for k, v in d.items():
            if not isinstance(v, np.ndarray) or v.size <= 0:
                continue
            if want == "cube" and v.ndim >= 3:
                cand.append((v.size, k, v))
            if want == "gt" and v.ndim >= 2:
                cand.append((v.size, k, v))
        if not cand:
            raise KeyError(f"no suitable array in {path.name}")
        cand.sort(key=lambda x: x[0], reverse=True)
        return np.asarray(cand[0][2])

    import h5py  # type: ignore

    with h5py.File(str(path), "r") as f:
        if key:
            if key not in f:
                raise KeyError(f"key '{key}' not found in {path.name}")
            return np.array(f[key])
        cand = []
        for k in f.keys():
            v = f[k]
            if not hasattr(v, "shape"):
                continue
            shape = tuple(int(x) for x in v.shape)
            size = int(np.prod(shape)) if shape else 0
            if size <= 0:
                continue
            if want == "cube" and len(shape) >= 3:
                cand.append((size, k))
            if want == "gt" and len(shape) >= 2:
                cand.append((size, k))
        if not cand:
            raise KeyError(f"no suitable dataset in {path.name}")
        cand.sort(key=lambda x: x[0], reverse=True)
        return np.array(f[cand[0][1]])


def _ensure_hwc(cube: np.ndarray) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got shape={cube.shape}")
    sh = list(cube.shape)
    bdim = int(np.argmin(sh))
    if bdim == 2:
        return cube
    if bdim == 0:
        return np.transpose(cube, (1, 2, 0))
    return np.transpose(cube, (0, 2, 1))


def _ensure_hw(gt: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    gt = np.asarray(gt)
    if gt.ndim == 3 and 1 in gt.shape:
        gt = np.squeeze(gt)
    if gt.ndim != 2:
        raise ValueError(f"gt must be 2D, got shape={gt.shape}")
    h, w = hw
    if gt.shape == (h, w):
        return gt
    if gt.shape == (w, h):
        return gt.T
    raise ValueError(f"gt shape {gt.shape} does not match cube hw {(h, w)}")


def _load_from_dataset_cfg(dataset: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str, int, int]:
    dataset_name = str(dataset.get("name", dataset.get("dataset", "dataset"))).strip()
    label_offset = int(dataset.get("label_offset", 1))
    num_classes = int(dataset.get("num_classes", dataset.get("n_classes", 0)))
    if num_classes <= 0:
        raise ValueError("num_classes must be provided in dataset config")

    raw_dir = str(dataset.get("raw_dir", "")).strip()
    if not raw_dir:
        raise KeyError("raw_dir missing in dataset config")
    cube_file = str(dataset.get("cube_file", "")).strip()
    gt_file = str(dataset.get("gt_file", "")).strip()
    if not cube_file or not gt_file:
        raise KeyError("cube_file/gt_file missing in dataset config")

    cube_key = str(dataset.get("cube_key", "") or "").strip()
    gt_key = str(dataset.get("gt_key", "") or "").strip()

    cube_path = _resolve_path(Path(raw_dir) / cube_file)
    gt_path = _resolve_path(Path(raw_dir) / gt_file)

    if not cube_path.exists():
        raise FileNotFoundError(f"cube file not found: {cube_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"gt file not found: {gt_path}")

    if cube_path.suffix.lower() in [".npy", ".npz"]:
        cube = np.load(str(cube_path))
        if isinstance(cube, np.lib.npyio.NpzFile):
            raise ValueError(f"cube npz not supported: {cube_path}")
    else:
        cube = _load_mat(cube_path, cube_key, want="cube")

    if gt_path.suffix.lower() in [".npy", ".npz"]:
        gt = np.load(str(gt_path))
        if isinstance(gt, np.lib.npyio.NpzFile):
            raise ValueError(f"gt npz not supported: {gt_path}")
    else:
        gt = _load_mat(gt_path, gt_key, want="gt")

    cube = _ensure_hwc(cube).astype(np.float32, copy=False)
    gt = _ensure_hw(gt, (int(cube.shape[0]), int(cube.shape[1]))).astype(np.int64, copy=False)

    return cube, gt, dataset_name, label_offset, num_classes


def _flat_indices(indices: np.ndarray, H: int, W: int) -> np.ndarray:
    idx = np.asarray(indices)
    if idx.ndim == 1:
        return idx.astype(np.int64, copy=False)
    if idx.ndim == 2 and idx.shape[1] == 2:
        r = idx[:, 0].astype(np.int64, copy=False)
        c = idx[:, 1].astype(np.int64, copy=False)
        return (r * W + c).astype(np.int64, copy=False)
    raise ValueError(f"indices must be (N,) or (N,2), got {idx.shape}")


def _fit_pca_train_only(
    cube_hwb: np.ndarray,
    train_flat_idx: np.ndarray,
    n_components: int,
    *,
    whiten: bool,
    random_state: int,
) -> Any:
    from sklearn.decomposition import PCA  # type: ignore

    _H, _W, B = cube_hwb.shape
    flat = cube_hwb.reshape(-1, B).astype(np.float32, copy=False)
    train_flat_idx = np.asarray(train_flat_idx, dtype=np.int64).reshape(-1)
    if train_flat_idx.size == 0:
        raise ValueError("empty train indices for PCA")
    X = flat[train_flat_idx]
    pca = PCA(
        n_components=int(n_components),
        whiten=bool(whiten),
        svd_solver="randomized",
        random_state=int(random_state),
    )
    pca.fit(X)
    return pca


def _apply_pca(cube_hwb: np.ndarray, pca: Any) -> np.ndarray:
    H, W, B = cube_hwb.shape
    flat = cube_hwb.reshape(-1, B).astype(np.float32, copy=False)
    Z = pca.transform(flat).astype(np.float32, copy=False)
    return Z.reshape(H, W, -1)


def _compute_class_weights_from_train(
    gt: np.ndarray,
    train_indices: np.ndarray,
    label_offset: int,
    num_classes: int,
) -> np.ndarray:
    y = gt.reshape(-1)[train_indices].astype(np.int64) - int(label_offset)
    y = y[(y >= 0) & (y < int(num_classes))]
    cnt = np.bincount(y, minlength=int(num_classes)).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    w = cnt.sum() / (float(num_classes) * cnt)
    w = w / w.mean()
    return w.astype(np.float32)


def _resolve_baseline_cfg(dataset_name: str) -> Path:
    key = "pu" if dataset_name == "pavia_university" else dataset_name
    return repo_root / "configs" / "baselines" / f"{key}.yaml"


def _install_optional_import_stubs() -> None:
    if "thop" not in sys.modules:
        m = types.ModuleType("thop")

        def _profile_stub(*_args: Any, **_kwargs: Any) -> Tuple[float, float]:
            raise RuntimeError("thop is optional and only used for profiling.")

        setattr(m, "profile", _profile_stub)
        sys.modules["thop"] = m

    if "torchsummaryX" not in sys.modules:
        m = types.ModuleType("torchsummaryX")

        def _summary_stub(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("torchsummaryX is optional and only used for model summary.")

        setattr(m, "summary", _summary_stub)
        sys.modules["torchsummaryX"] = m

    if "timm.models.vision_transformer" not in sys.modules:
        timm = sys.modules.get("timm", types.ModuleType("timm"))
        models = getattr(timm, "models", None)
        if models is None:
            models = types.ModuleType("timm.models")
            setattr(timm, "models", models)
        vt = types.ModuleType("timm.models.vision_transformer")
        setattr(vt, "_cfg", lambda **_kwargs: {})
        setattr(models, "vision_transformer", vt)
        sys.modules["timm"] = timm
        sys.modules["timm.models"] = models
        sys.modules["timm.models.vision_transformer"] = vt


def _load_external_gscvit_class() -> Any:
    try:
        import einops  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("GSC-ViT requires 'einops'. Install it with: pip install einops") from e

    _install_optional_import_stubs()

    model_path = repo_root / "external" / "baselines" / "gsc_vit" / "models" / "gscvit.py"
    if not model_path.exists():
        raise FileNotFoundError(f"GSC-ViT model file not found: {model_path}")

    spec = importlib.util.spec_from_file_location("gsc_vit_external_model", str(model_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create module spec for {model_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, "GSCViT"):
        raise AttributeError(f"GSCViT class not found in {model_path}")
    return mod.GSCViT


def _to_int_tuple(v: Any, name: str) -> Tuple[int, ...]:
    if isinstance(v, (list, tuple)):
        out = tuple(int(x) for x in v)
    else:
        out = (int(v),)
    if len(out) == 0:
        raise ValueError(f"{name} must not be empty")
    return out


def _broadcast_tuple(v: Tuple[int, ...], n: int, name: str) -> Tuple[int, ...]:
    if len(v) == 1 and n > 1:
        return tuple([int(v[0])] * n)
    if len(v) != n:
        raise ValueError(f"{name} length must be {n} (or 1 for broadcast), got {len(v)}")
    return tuple(int(x) for x in v)


class _GSCViTWrapper(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"GSC-ViT expects x as (N,Bands,P,P), got shape={tuple(x.shape)}")
        return self.base(x.unsqueeze(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--train_cfg", required=True)  # compatibility; not used
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)

    ap.add_argument("--baseline", default="gsc_vit")
    ap.add_argument("--patch_size", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_amp", action="store_true")
    args = ap.parse_args()

    set_global_seed(int(args.seed))

    dataset = load_yaml(args.dataset_cfg)
    split = load_json(args.split_json)

    cube, gt, dataset_name, label_offset, num_classes = _load_from_dataset_cfg(dataset)
    H, W = int(gt.shape[0]), int(gt.shape[1])

    train_indices = np.unique(_flat_indices(np.asarray(split["train_indices"]), H, W).astype(np.int64, copy=False))
    val_indices = np.unique(_flat_indices(np.asarray(split["val_indices"]), H, W).astype(np.int64, copy=False))
    test_indices = np.unique(_flat_indices(np.asarray(split["test_indices"]), H, W).astype(np.int64, copy=False))

    _assert_disjoint(train_indices, val_indices, "train", "val")
    _assert_disjoint(train_indices, test_indices, "train", "test")
    _assert_disjoint(val_indices, test_indices, "val", "test")

    bcfg_path = _resolve_baseline_cfg(dataset_name)
    if not bcfg_path.exists():
        raise FileNotFoundError(f"GSC-ViT baseline yaml not found: {bcfg_path}")
    bcfg = load_yaml(str(bcfg_path))
    baseline_cfg_sha1 = _sha1_file(bcfg_path)

    gsc_cfg = _get(bcfg, ("baseline", "gsc_vit"), None)
    if gsc_cfg is None and isinstance(bcfg, dict):
        gsc_cfg = bcfg.get("gsc_vit", bcfg.get("gscvit", bcfg.get("gsc-vit")))
    if not isinstance(gsc_cfg, dict):
        raise KeyError(f"Cannot find baseline.gsc_vit in {bcfg_path}")

    def _hp(name: str, default: Any) -> Any:
        return _get(gsc_cfg, (name,), default)

    yaml_patch_size = int(_hp("patch_size", 15))
    if yaml_patch_size != 15:
        raise ValueError("baseline.gsc_vit.patch_size must be 15")
    if int(args.patch_size) != yaml_patch_size:
        raise ValueError(f"CLI patch_size={args.patch_size} != yaml patch_size={yaml_patch_size}")
    patch_size = yaml_patch_size

    pca_bands = int(_hp("pca_bands", 0))
    pca_whiten = bool(_hp("pca_whiten", False))

    dims = _to_int_tuple(_hp("dims", [256, 128, 64, 32]), "dims")
    depth = _to_int_tuple(_hp("depth", [1, 1, 1]), "depth")
    stage_count = len(depth)

    heads = _broadcast_tuple(_to_int_tuple(_hp("heads", [1, 1, 1]), "heads"), stage_count, "heads")
    group_spatial_size = _broadcast_tuple(
        _to_int_tuple(_hp("group_spatial_size", [5, 5, 5]), "group_spatial_size"),
        stage_count,
        "group_spatial_size",
    )
    padding = _broadcast_tuple(_to_int_tuple(_hp("padding", [1, 1, 1]), "padding"), stage_count, "padding")
    num_groups = _broadcast_tuple(_to_int_tuple(_hp("num_groups", [16, 16, 16]), "num_groups"), stage_count, "num_groups")

    if len(dims) != stage_count + 1:
        raise ValueError(f"dims length must be stage_count + 1 (= {stage_count + 1}), got {len(dims)}")

    for i, gss in enumerate(group_spatial_size):
        if gss <= 0:
            raise ValueError(f"group_spatial_size[{i}] must be > 0")
        if patch_size % int(gss) != 0:
            raise ValueError(f"patch_size={patch_size} must be divisible by group_spatial_size[{i}]={gss}")

    for i, ng in enumerate(num_groups):
        if ng <= 0:
            raise ValueError(f"num_groups[{i}] must be > 0")
        dim_in = int(dims[i])
        dim_out = int(dims[i + 1])
        if dim_in % int(ng) != 0 or dim_out % int(ng) != 0:
            raise ValueError(
                f"GSC stage[{i}] channels must be divisible by num_groups[{i}]={ng}, "
                f"got dim_in={dim_in}, dim_out={dim_out}"
            )

    dropout = float(_hp("dropout", 0.1))

    batch_size = int(_hp("batch_size", 64))
    eval_batch_size = int(_hp("eval_batch_size", 256))
    num_workers = int(_hp("num_workers", 0))

    lr = float(_hp("lr", 5e-4))
    weight_decay = float(_hp("weight_decay", 1e-2))
    betas = _hp("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("baseline.gsc_vit.betas must be a list of length 2")
    betas_tuple = (float(betas[0]), float(betas[1]))
    grad_clip = float(_hp("grad_clip", 1.0))

    max_epochs = int(_hp("max_epochs", 180))
    early_stop_patience = int(_hp("early_stop_patience", 20))
    min_epochs = int(_hp("min_epochs", 40))
    select_metric = str(_hp("select_metric", "kappa")).lower().strip()

    scheduler_name = str(_hp("scheduler", "cosine")).lower().strip()
    min_lr = float(_hp("min_lr", 1e-6))

    augment = bool(_hp("augment", False))
    label_smoothing = float(_hp("label_smoothing", 0.0))
    spectral_dropout = float(_hp("spectral_dropout", 0.0))
    input_noise_std = float(_hp("input_noise_std", 0.0))

    class_weight = str(_hp("class_weight", "null")).lower().strip()
    class_weight_power = float(_hp("class_weight_power", 0.0))

    norm_mean_global_blend = float(_hp("norm_mean_global_blend", 0.0))
    norm_std_global_ratio = float(_hp("norm_std_global_ratio", 0.05))
    norm_std_abs_floor = float(_hp("norm_std_abs_floor", 1e-3))

    cfg_use_amp = bool(_hp("use_amp", False))
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = bool((args.use_amp or cfg_use_amp) and device.type == "cuda")

    if select_metric not in {"kappa", "oa", "loss"}:
        raise ValueError("select_metric must be one of: kappa / oa / loss")
    if pca_bands < 0:
        raise ValueError("pca_bands must be >= 0")

    raw_bands = int(cube.shape[-1])
    cube_used = cube.astype(np.float32, copy=False)

    spectral_method = "raw"
    spectral_meta: Dict[str, Any] = {}

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(str(ckpt_dir))
    meta_dir = out_dir / "meta"
    ensure_dir(str(meta_dir))

    pca_path = meta_dir / "pca_params.npz"
    if pca_bands > 0:
        if pca_bands > raw_bands:
            raise ValueError(f"pca_bands={pca_bands} exceeds raw bands={raw_bands}")
        spectral_method = "pca"
        if pca_path.exists():
            npz = np.load(str(pca_path))
            pca_mean = npz["mean"].astype(np.float32)
            pca_comps = npz["comps"].astype(np.float32)
            spectral_meta = {"pca_k": int(pca_comps.shape[0]), "pca_whiten": bool(pca_whiten), "solver": "cached"}
        else:
            pca = _fit_pca_train_only(
                cube_used,
                train_indices,
                pca_bands,
                whiten=pca_whiten,
                random_state=int(args.seed),
            )
            pca_mean = pca.mean_.astype(np.float32, copy=False)
            pca_comps = pca.components_.astype(np.float32, copy=False)
            np.savez_compressed(
                pca_path,
                mean=pca_mean,
                comps=pca_comps,
            )
            spectral_meta = {"pca_k": int(pca_bands), "pca_whiten": bool(pca_whiten), "solver": "randomized"}

        class _SimplePCA:
            def __init__(self, mean: np.ndarray, comps: np.ndarray) -> None:
                self.mean = mean
                self.comps = comps

            def transform(self, x: np.ndarray) -> np.ndarray:
                x = x.astype(np.float32, copy=False)
                return (x - self.mean.reshape(1, -1)) @ self.comps.T

        cube_used = _apply_pca(cube_used, _SimplePCA(pca_mean, pca_comps))

    bands = int(cube_used.shape[-1])

    norm_path = meta_dir / "norm_stats.npz"
    if norm_path.exists():
        npz = np.load(str(norm_path))
        mean = npz["mean"].astype(np.float32)
        std = npz["std"].astype(np.float32)
    else:
        mean, std = compute_train_norm(
            cube_used,
            train_indices,
            mean_global_blend=norm_mean_global_blend,
            std_global_ratio=norm_std_global_ratio,
            std_abs_floor=norm_std_abs_floor,
        )
        np.savez_compressed(norm_path, mean=mean, std=std)

    g = torch.Generator()
    g.manual_seed(int(args.seed))

    ds_tr = HSIPatchDataset(
        cube=cube_used,
        gt=gt,
        indices=train_indices,
        patch_size=patch_size,
        label_offset=label_offset,
        mean=mean,
        std=std,
        augment=augment,
    )
    ds_va = HSIPatchDataset(
        cube=cube_used,
        gt=gt,
        indices=val_indices,
        patch_size=patch_size,
        label_offset=label_offset,
        mean=mean,
        std=std,
        augment=False,
    )
    ds_te = HSIPatchDataset(
        cube=cube_used,
        gt=gt,
        indices=test_indices,
        patch_size=patch_size,
        label_offset=label_offset,
        mean=mean,
        std=std,
        augment=False,
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    GSCViT = _load_external_gscvit_class()
    base_model = GSCViT(
        num_classes=num_classes,
        depth=tuple(depth),
        heads=tuple(heads),
        group_spatial_size=list(group_spatial_size),
        channels=bands,
        dropout=dropout,
        padding=list(padding),
        dims=tuple(dims),
        num_groups=list(num_groups),
    )
    model = _GSCViTWrapper(base_model).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas_tuple)

    scheduler = None
    if scheduler_name in {"cos", "cosine"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs, eta_min=min_lr)
    elif scheduler_name in {"none", "off", ""}:
        scheduler = None
    else:
        raise ValueError(f"unsupported scheduler='{scheduler_name}' for GSC-ViT")

    cw_tensor = None
    if class_weight in {"balanced", "balance"} and class_weight_power > 0:
        w = _compute_class_weights_from_train(gt, train_indices, label_offset, num_classes)
        alpha = float(np.clip(class_weight_power, 0.0, 1.0))
        w = (1.0 - alpha) * np.ones_like(w, dtype=np.float32) + alpha * w
        cw_tensor = torch.tensor(w, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=cw_tensor, label_smoothing=label_smoothing)

    spec_dropout = nn.Dropout2d(p=spectral_dropout) if spectral_dropout > 0 else None
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_score = -1e18
    best_ep = -1
    bad = 0
    best_path = ckpt_dir / "best.pt"

    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        ep_loss = 0.0
        n_seen = 0

        for x, _x_spec, y in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                x_in = x
                if spec_dropout is not None:
                    x_in = spec_dropout(x_in)
                if input_noise_std > 0:
                    x_in = x_in + torch.randn_like(x_in) * input_noise_std
                logits = model(x_in)

            loss = criterion(logits.float(), y)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optim)
            scaler.update()

            bs_now = int(x.shape[0])
            ep_loss += float(loss.detach().item()) * bs_now
            n_seen += bs_now

        ep_loss = ep_loss / max(1, n_seen)
        if scheduler is not None:
            scheduler.step()

        val = _evaluate(model, dl_va, device, num_classes, criterion=criterion, use_amp=use_amp)

        if select_metric == "kappa":
            score = float(val.get("Kappa", 0.0))
        elif select_metric == "oa":
            score = float(val.get("OA", 0.0))
        else:
            score = -float(val.get("loss", 1e18))

        improved = score > best_score + 1e-12
        if improved:
            best_score = score
            best_ep = ep
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": {
                        "dataset": dataset_name,
                        "baseline": "gsc_vit",
                        "seed": int(args.seed),
                        "patch_size": int(patch_size),
                        "bands": int(bands),
                        "raw_bands": int(raw_bands),
                        "spectral_method": spectral_method,
                        "spectral_meta": spectral_meta,
                        "num_classes": int(num_classes),
                        "label_offset": int(label_offset),
                        "dims": list(int(x) for x in dims),
                        "depth": list(int(x) for x in depth),
                        "heads": list(int(x) for x in heads),
                        "group_spatial_size": list(int(x) for x in group_spatial_size),
                        "padding": list(int(x) for x in padding),
                        "num_groups": list(int(x) for x in num_groups),
                        "dropout": float(dropout),
                        "baseline_cfg": str(bcfg_path),
                        "baseline_cfg_sha1": baseline_cfg_sha1,
                        "select_metric": select_metric,
                        "class_weight": class_weight,
                        "class_weight_power": float(class_weight_power),
                        "label_smoothing": float(label_smoothing),
                        "spectral_dropout": float(spectral_dropout),
                        "input_noise_std": float(input_noise_std),
                        "augment": bool(augment),
                    },
                },
                best_path,
            )
        else:
            if ep >= min_epochs:
                bad += 1

        if early_stop_patience > 0 and bad >= early_stop_patience:
            break

        if ep % 10 == 0 or improved:
            dt = time.time() - t0
            vloss = val.get("loss", float("nan"))
            print(
                f"[ep {ep:04d}] loss={ep_loss:.6f} "
                f"| VAL OA={val['OA']:.4f} AA={val['AA']:.4f} Kappa={val['Kappa']:.4f} "
                f"| VAL loss={vloss:.6f} "
                f"| score={score:.6f} best={best_score:.6f}@{best_ep} bad={bad} "
                f"| t={dt:.1f}s "
                f"| spectral={spectral_method} bands={bands}"
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    val_best = _evaluate(model, dl_va, device, num_classes, criterion=criterion, use_amp=use_amp)
    test_best = _evaluate(model, dl_te, device, num_classes, criterion=criterion, use_amp=use_amp)

    metrics = {
        "VAL": val_best,
        "TEST": test_best,
        "meta": {
            "dataset": dataset_name,
            "baseline": "gsc_vit",
            "split_json": str(Path(args.split_json)),
            "ckpt": str(best_path),
            "ckpt_key": "model",
            "label_offset": int(label_offset),
            "num_classes": int(num_classes),
            "patch_size": int(patch_size),
            "bands": int(bands),
            "raw_bands": int(raw_bands),
            "spectral_method": spectral_method,
            "spectral_meta": spectral_meta,
            "norm_path": str(norm_path),
            "dims": list(int(x) for x in dims),
            "depth": list(int(x) for x in depth),
            "heads": list(int(x) for x in heads),
            "group_spatial_size": list(int(x) for x in group_spatial_size),
            "padding": list(int(x) for x in padding),
            "num_groups": list(int(x) for x in num_groups),
            "dropout": float(dropout),
            "baseline_cfg": str(bcfg_path),
            "baseline_cfg_sha1": baseline_cfg_sha1,
            "best_ep": int(best_ep),
            "select_metric": select_metric,
            "class_weight": class_weight,
            "class_weight_power": float(class_weight_power),
            "label_smoothing": float(label_smoothing),
            "spectral_dropout": float(spectral_dropout),
            "input_noise_std": float(input_noise_std),
            "augment": bool(augment),
            "scheduler": scheduler_name,
            "use_amp": bool(use_amp),
        },
    }
    save_json(out_dir / "metrics.json", metrics)
    print(f"[done] out_dir={out_dir}")


if __name__ == "__main__":
    main()
