from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

_NORM_PAD_CACHE: "OrderedDict[Tuple[Any, ...], np.ndarray]" = OrderedDict()
_NORM_PAD_CACHE_MAX = 4


def _norm_pad_cache_key(
    cube: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    half: int,
    pad_mode: str,
) -> Tuple[Any, ...]:
    ptr = int(cube.__array_interface__["data"][0])
    shape = tuple(int(x) for x in cube.shape)
    strides = tuple(int(x) for x in cube.strides)
    mean_key = np.asarray(mean, dtype=np.float32).reshape(-1).tobytes()
    std_key = np.asarray(std, dtype=np.float32).reshape(-1).tobytes()
    return (ptr, shape, strides, int(half), str(pad_mode), mean_key, std_key)


def _get_or_build_norm_padded_cube(
    cube: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    half: int,
    pad_mode: str,
) -> np.ndarray:
    key = _norm_pad_cache_key(cube, mean, std, half, pad_mode)
    got = _NORM_PAD_CACHE.get(key)
    if got is not None:
        _NORM_PAD_CACHE.move_to_end(key)
        return got

    cube_norm = np.ascontiguousarray((cube - mean) / std, dtype=np.float32)
    cube_pad = np.pad(cube_norm, ((half, half), (half, half), (0, 0)), mode=str(pad_mode))
    cube_pad = np.ascontiguousarray(cube_pad, dtype=np.float32)
    _NORM_PAD_CACHE[key] = cube_pad
    _NORM_PAD_CACHE.move_to_end(key)
    while len(_NORM_PAD_CACHE) > int(_NORM_PAD_CACHE_MAX):
        _NORM_PAD_CACHE.popitem(last=False)
    return cube_pad


def _ensure_hwb_cube(cube: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the cube in `(H, W, B)` order and the label map in `(H, W)`."""
    cube = np.asarray(cube)
    gt = np.asarray(gt)

    if cube.ndim != 3:
        raise ValueError(f"[HSIPatchDataset] cube must be 3D, got shape={cube.shape}")
    if gt.ndim != 2:
        raise ValueError(f"[HSIPatchDataset] gt must be 2D, got shape={gt.shape}")

    H, W = gt.shape

    if cube.shape[0] == H and cube.shape[1] == W:
        return cube, gt

    if cube.shape[1] == H and cube.shape[2] == W:
        cube = np.transpose(cube, (1, 2, 0))
        return cube, gt

    if cube.shape[0] == W and cube.shape[1] == H:
        cube = np.transpose(cube, (1, 0, 2))
        return cube, gt

    raise ValueError(
        "[HSIPatchDataset] cube/gt shape mismatch. "
        f"gt(H,W)={gt.shape}, cube={cube.shape}. "
        "Expected cube (H,W,B) or (B,H,W) or (W,H,B)."
    )


def _to_flat_indices(indices: np.ndarray, gt_shape: Tuple[int, int]) -> np.ndarray:
    """Convert flat indices or `(row, col)` pairs to flat indices."""
    indices = np.asarray(indices)
    _, W = gt_shape

    if indices.ndim == 1:
        return indices.astype(np.int64, copy=False)

    if indices.ndim == 2 and indices.shape[1] == 2:
        r = indices[:, 0].astype(np.int64, copy=False)
        c = indices[:, 1].astype(np.int64, copy=False)
        return (r * W + c).astype(np.int64, copy=False)

    raise ValueError(f"[HSIPatchDataset] indices must be (N,) or (N,2), got shape={indices.shape}")


@dataclass
class HSIPatchDataset(Dataset):
    cube: np.ndarray
    gt: np.ndarray
    indices: np.ndarray
    patch_size: int
    mean: np.ndarray
    std: np.ndarray
    label_offset: int = 1
    augment: bool = False
    return_x_spec: bool = True

    noise_std: float = 0.0
    pad_mode: str = "edge"

    def __post_init__(self) -> None:
        ps = int(self.patch_size)
        if ps <= 0:
            raise ValueError(f"[HSIPatchDataset] patch_size must be >0, got {ps}")
        if ps % 2 != 1:
            raise ValueError(f"[HSIPatchDataset] patch_size must be odd, got {ps}")
        self.patch_size = ps
        self.half = ps // 2

        self.cube, self.gt = _ensure_hwb_cube(self.cube, self.gt)

        self.cube = self.cube.astype(np.float32, copy=False)
        self.gt = self.gt.astype(np.int64, copy=False)

        self.h, self.w, self.b = self.cube.shape

        flat_idx = _to_flat_indices(self.indices, (self.h, self.w)).reshape(-1).astype(np.int64, copy=False)
        if flat_idx.size == 0:
            raise ValueError("[HSIPatchDataset] empty indices")

        mn = int(flat_idx.min())
        mx = int(flat_idx.max())
        if mn < 0 or mx >= self.h * self.w:
            raise ValueError(
                f"[HSIPatchDataset] indices out of range: min={mn}, max={mx}, "
                f"valid=[0, {self.h*self.w-1}] (H={self.h}, W={self.w}). "
                "Your split_json may be inconsistent with cube/gt layout."
            )
        self.indices = flat_idx

        self.mean = np.asarray(self.mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.asarray(self.std, dtype=np.float32).reshape(1, 1, -1)
        if self.mean.shape[-1] != self.b:
            raise ValueError(f"[HSIPatchDataset] mean bands mismatch: mean={self.mean.shape[-1]} vs cube_bands={self.b}")
        if self.std.shape[-1] != self.b:
            raise ValueError(f"[HSIPatchDataset] std bands mismatch: std={self.std.shape[-1]} vs cube_bands={self.b}")
        self.std = np.maximum(self.std, 1e-6).astype(np.float32)

        self.noise_std = float(self.noise_std)
        self.return_x_spec = bool(self.return_x_spec)
        pad_mode = str(self.pad_mode or "edge").strip().lower()
        if pad_mode in ("replicate", "nearest"):
            pad_mode = "edge"
        if pad_mode not in {"edge", "reflect", "symmetric"}:
            raise ValueError(f"[HSIPatchDataset] unsupported pad_mode='{self.pad_mode}'")
        self.pad_mode = pad_mode

        self._cube_pad = _get_or_build_norm_padded_cube(
            self.cube,
            self.mean,
            self.std,
            self.half,
            self.pad_mode,
        )

    def __len__(self) -> int:
        return int(self.indices.size)

    def _spatial_aug(self, patch: np.ndarray) -> np.ndarray:
        """Apply random flips and right-angle rotations."""
        if np.random.rand() < 0.5:
            patch = patch[::-1, :, :]
        if np.random.rand() < 0.5:
            patch = patch[:, ::-1, :]
        k = int(np.random.randint(0, 4))
        if k:
            patch = np.rot90(patch, k, axes=(0, 1))
        patch = np.ascontiguousarray(patch)
        return patch

    def _spectral_aug(self, patch: np.ndarray) -> np.ndarray:
        if self.noise_std > 0.0:
            noise = np.random.normal(0.0, self.noise_std, size=patch.shape).astype(np.float32)
            patch = patch + noise
        return patch

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        r = idx // self.w
        c = idx % self.w

        y_raw = int(self.gt[r, c])
        y = y_raw - int(self.label_offset)
        if y < 0:
            raise ValueError(
                f"[HSIPatchDataset] got background/invalid label at (r,c)=({r},{c}), "
                f"gt={y_raw}, label_offset={self.label_offset}. "
                "Your split should not include background pixels."
            )

        if self._cube_pad is not None:
            patch = self._cube_pad[r : r + self.patch_size, c : c + self.patch_size, :]
        else:
            r0 = max(0, r - self.half)
            r1 = min(self.h, r + self.half + 1)
            c0 = max(0, c - self.half)
            c1 = min(self.w, c + self.half + 1)

            patch = self.cube[r0:r1, c0:c1, :]

            pad_top = max(0, self.half - r)
            pad_left = max(0, self.half - c)
            pad_bottom = max(0, (r + self.half + 1) - self.h)
            pad_right = max(0, (c + self.half + 1) - self.w)

            if pad_top or pad_bottom or pad_left or pad_right:
                patch = np.pad(
                    patch,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode=self.pad_mode,
                )

            if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                patch = patch[: self.patch_size, : self.patch_size, :]

            patch = (patch - self.mean) / self.std

        if self.augment:
            patch = self._spatial_aug(patch)
            patch = self._spectral_aug(patch)

        patch = np.ascontiguousarray(patch, dtype=np.float32)

        x_np = np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32)
        x = torch.from_numpy(x_np)

        y_t = torch.tensor(int(y), dtype=torch.long)
        if self.return_x_spec:
            x_spec = x[:, self.half, self.half].contiguous()
            return x, x_spec, y_t
        return x, y_t


def compute_train_norm(
    cube: np.ndarray,
    train_indices: np.ndarray,
    *,
    mean_global_blend: float = 0.0,
    std_global_ratio: float = 0.0,
    std_abs_floor: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-band normalization statistics from the training split."""
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"[compute_train_norm] cube must be 3D, got {cube.shape}")

    if cube.shape[0] <= 256 and cube.shape[-1] > 256:
        cube = np.transpose(cube, (1, 2, 0))

    H, W, B = cube.shape
    flat = cube.reshape(-1, B).astype(np.float64, copy=False)

    idx = np.asarray(train_indices, dtype=np.int64)
    if idx.ndim == 2 and idx.shape[1] == 2:
        idx = (idx[:, 0] * W + idx[:, 1]).astype(np.int64, copy=False)
    else:
        idx = idx.reshape(-1).astype(np.int64, copy=False)

    if idx.size == 0:
        raise ValueError("[compute_train_norm] empty train_indices")

    mn = int(idx.min())
    mx = int(idx.max())
    if mn < 0 or mx >= H * W:
        raise ValueError(
            f"[compute_train_norm] train_indices out of range: min={mn}, max={mx}, valid=[0, {H*W-1}]"
        )

    train_vals = flat[idx]
    train_mean = train_vals.mean(axis=0).astype(np.float32)
    global_mean = flat.mean(axis=0).astype(np.float32)
    w = float(mean_global_blend)
    w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
    mean = ((1.0 - w) * train_mean + w * global_mean).astype(np.float32)

    train_std = train_vals.std(axis=0).astype(np.float32)
    global_std = flat.std(axis=0).astype(np.float32)
    r = float(std_global_ratio)
    r = 0.0 if r < 0.0 else r
    std = np.maximum(train_std, r * global_std).astype(np.float32)
    std = np.maximum(std, float(std_abs_floor)).astype(np.float32)

    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std
