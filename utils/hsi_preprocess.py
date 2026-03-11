from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _ensure_hwb(cube: np.ndarray, gt_shape: Tuple[int, int] | None = None) -> np.ndarray:
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"[hsi_preprocess] cube must be 3D, got {cube.shape}")

    if gt_shape is None:
        return np.ascontiguousarray(cube, dtype=np.float32)

    h, w = int(gt_shape[0]), int(gt_shape[1])
    if cube.shape[0] == h and cube.shape[1] == w:
        return np.ascontiguousarray(cube, dtype=np.float32)
    if cube.shape[1] == h and cube.shape[2] == w:
        return np.ascontiguousarray(np.transpose(cube, (1, 2, 0)), dtype=np.float32)
    raise ValueError(
        f"[hsi_preprocess] cube/gt shape mismatch: cube={cube.shape}, gt={gt_shape}. "
        "Expected cube in (H,W,B) or (B,H,W)."
    )


def resolve_spectral_preprocess(train_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg = dict(train_cfg or {})
    mode = str(cfg.get("spectral_preprocess", "none")).strip().lower()
    if mode in ("", "none", "raw", "identity"):
        return {"mode": "none"}
    if mode != "pca":
        raise ValueError(f"Unsupported spectral_preprocess='{mode}'. Supported: none, pca.")

    pca_bands = int(cfg.get("spectral_pca_bands", cfg.get("pca_bands", 0)))
    if pca_bands <= 0:
        raise ValueError("spectral_preprocess='pca' requires spectral_pca_bands > 0.")

    return {
        "mode": "pca",
        "pca_bands": int(pca_bands),
        "pca_whiten": bool(cfg.get("spectral_pca_whiten", cfg.get("pca_whiten", False))),
        "chunk_pixels": int(max(1024, cfg.get("spectral_chunk_pixels", 65536))),
    }


def _to_flat_indices(indices: np.ndarray, gt_shape: Tuple[int, int]) -> np.ndarray:
    indices = np.asarray(indices)
    _, w = gt_shape
    if indices.ndim == 1:
        return indices.astype(np.int64, copy=False).reshape(-1)
    if indices.ndim == 2 and indices.shape[1] == 2:
        r = indices[:, 0].astype(np.int64, copy=False)
        c = indices[:, 1].astype(np.int64, copy=False)
        return (r * w + c).astype(np.int64, copy=False)
    raise ValueError(f"[hsi_preprocess] indices must be (N,) or (N,2), got {indices.shape}")


def _fit_pca_state(
    cube: np.ndarray,
    train_indices: np.ndarray,
    *,
    pca_bands: int,
    pca_whiten: bool,
    chunk_pixels: int,
) -> Dict[str, Any]:
    h, w, bands = cube.shape
    flat_idx = _to_flat_indices(train_indices, (h, w))
    if flat_idx.size <= 1:
        raise ValueError("[hsi_preprocess] PCA needs at least 2 train samples.")

    x_train = cube.reshape(-1, bands)[flat_idx].astype(np.float32, copy=False)
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    x_center = x_train - mean[None, :]
    _, s, vt = np.linalg.svd(x_center, full_matrices=False)

    max_rank = int(vt.shape[0])
    out_bands = int(min(max_rank, max(1, int(pca_bands))))
    components = vt[:out_bands].astype(np.float32, copy=False)

    denom = max(1, int(x_train.shape[0]) - 1)
    explained_variance = ((s[:out_bands] ** 2) / float(denom)).astype(np.float32, copy=False)
    total_variance = float(np.maximum((s ** 2).sum() / float(denom), 1.0e-12))
    explained_ratio = (explained_variance / total_variance).astype(np.float32, copy=False)

    scale = np.ones((out_bands,), dtype=np.float32)
    if bool(pca_whiten):
        scale = np.sqrt(np.maximum(explained_variance, 1.0e-6)).astype(np.float32, copy=False)

    return {
        "mode": "pca",
        "raw_bands": int(bands),
        "out_bands": int(out_bands),
        "chunk_pixels": int(max(1024, int(chunk_pixels))),
        "pca_whiten": bool(pca_whiten),
        "mean": mean,
        "components": components,
        "scale": scale,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_ratio,
    }


def apply_spectral_preprocess(cube: np.ndarray, state: Dict[str, Any] | None) -> np.ndarray:
    mode = str((state or {}).get("mode", "none")).strip().lower()
    cube = np.ascontiguousarray(cube, dtype=np.float32)
    if mode in ("", "none", "raw", "identity"):
        return cube
    if mode != "pca":
        raise ValueError(f"Unsupported spectral preprocess mode='{mode}'.")

    h, w, bands = cube.shape
    mean = np.asarray(state["mean"], dtype=np.float32).reshape(1, -1)
    components = np.asarray(state["components"], dtype=np.float32)
    if components.ndim != 2 or components.shape[1] != bands:
        raise ValueError(
            f"[hsi_preprocess] PCA components shape mismatch: components={components.shape}, cube_bands={bands}"
        )

    scale = np.asarray(state.get("scale", np.ones((components.shape[0],), dtype=np.float32)), dtype=np.float32)
    if scale.ndim != 1 or scale.shape[0] != components.shape[0]:
        raise ValueError(
            f"[hsi_preprocess] PCA scale shape mismatch: scale={scale.shape}, out_bands={components.shape[0]}"
        )

    chunk_pixels = int(max(1024, int(state.get("chunk_pixels", 65536))))
    flat = cube.reshape(-1, bands)
    out = np.empty((flat.shape[0], components.shape[0]), dtype=np.float32)
    comp_t = np.ascontiguousarray(components.T, dtype=np.float32)
    for start in range(0, flat.shape[0], chunk_pixels):
        end = min(flat.shape[0], start + chunk_pixels)
        x = flat[start:end].astype(np.float32, copy=False)
        y = (x - mean) @ comp_t
        y = y / scale[None, :]
        out[start:end] = y.astype(np.float32, copy=False)
    return np.ascontiguousarray(out.reshape(h, w, components.shape[0]), dtype=np.float32)


def fit_and_apply_spectral_preprocess(
    cube: np.ndarray,
    train_indices: np.ndarray,
    train_cfg: Dict[str, Any] | None,
    *,
    gt_shape: Tuple[int, int] | None = None,
    save_path: str | Path | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    cube_hwb = _ensure_hwb(cube, gt_shape=gt_shape)
    spec = resolve_spectral_preprocess(train_cfg)
    if spec["mode"] == "none":
        return cube_hwb, {"mode": "none", "raw_bands": int(cube_hwb.shape[-1]), "out_bands": int(cube_hwb.shape[-1])}

    state = _fit_pca_state(
        cube_hwb,
        train_indices,
        pca_bands=int(spec["pca_bands"]),
        pca_whiten=bool(spec["pca_whiten"]),
        chunk_pixels=int(spec["chunk_pixels"]),
    )
    cube_out = apply_spectral_preprocess(cube_hwb, state)
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            mode=np.asarray(state["mode"]),
            raw_bands=np.asarray(int(state["raw_bands"]), dtype=np.int32),
            out_bands=np.asarray(int(state["out_bands"]), dtype=np.int32),
            chunk_pixels=np.asarray(int(state["chunk_pixels"]), dtype=np.int32),
            pca_whiten=np.asarray(int(bool(state["pca_whiten"])), dtype=np.int8),
            mean=np.asarray(state["mean"], dtype=np.float32),
            components=np.asarray(state["components"], dtype=np.float32),
            scale=np.asarray(state["scale"], dtype=np.float32),
            explained_variance=np.asarray(state["explained_variance"], dtype=np.float32),
            explained_variance_ratio=np.asarray(state["explained_variance_ratio"], dtype=np.float32),
        )
    return cube_out, state


def load_spectral_preprocess(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"spectral preprocess file not found: {p}")

    z = np.load(p, allow_pickle=False)
    mode = str(z["mode"].item()).strip().lower()
    if mode in ("", "none", "raw", "identity"):
        return {"mode": "none"}
    if mode != "pca":
        raise ValueError(f"Unsupported spectral preprocess mode='{mode}' in {p}")
    return {
        "mode": "pca",
        "raw_bands": int(np.asarray(z["raw_bands"]).item()),
        "out_bands": int(np.asarray(z["out_bands"]).item()),
        "chunk_pixels": int(np.asarray(z["chunk_pixels"]).item()),
        "pca_whiten": bool(int(np.asarray(z["pca_whiten"]).item())),
        "mean": np.asarray(z["mean"], dtype=np.float32),
        "components": np.asarray(z["components"], dtype=np.float32),
        "scale": np.asarray(z["scale"], dtype=np.float32),
        "explained_variance": np.asarray(z["explained_variance"], dtype=np.float32),
        "explained_variance_ratio": np.asarray(z["explained_variance_ratio"], dtype=np.float32),
    }
