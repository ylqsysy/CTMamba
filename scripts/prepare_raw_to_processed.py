#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare raw .mat -> processed .npy (cube.npy / gt.npy)

Supports:
- MATLAB < v7.3 via scipy.io.loadmat
- MATLAB v7.3 (HDF5) via h5py

Outputs:
data/processed/<dataset>/raw/cube.npy  (H,W,B) float32
data/processed/<dataset>/raw/gt.npy    (H,W)   int64
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np

from hsi3d.utils.io import load_yaml, ensure_dir


def _is_numeric_ndarray(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.dtype.kind in ("i", "u", "f")


def _load_mat_scipy(path: Path) -> Dict[str, Any]:
    from scipy.io import loadmat  # lazy import
    d = loadmat(str(path))
    # filter matlab metadata keys
    out = {}
    for k, v in d.items():
        if k.startswith("__"):
            continue
        out[k] = v
    return out


def _read_h5_dataset(obj) -> np.ndarray:
    import h5py

    if isinstance(obj, h5py.Dataset):
        arr = np.array(obj)
        return arr
    raise TypeError(f"Unsupported HDF5 object type: {type(obj)}")


def _load_mat_h5py(path: Path) -> Dict[str, Any]:
    import h5py

    out: Dict[str, Any] = {}
    with h5py.File(str(path), "r") as f:
        for k in f.keys():
            if k.startswith("#"):
                continue
            obj = f[k]
            try:
                out[k] = _read_h5_dataset(obj)
            except Exception:
                # ignore unsupported groups/refs in this lightweight loader
                pass
    return out


def load_mat_any(path: Path) -> Dict[str, Any]:
    # try scipy first, fallback to h5py for v7.3
    try:
        return _load_mat_scipy(path)
    except NotImplementedError:
        return _load_mat_h5py(path)
    except Exception:
        # if scipy fails for other reasons, also try h5py
        try:
            return _load_mat_h5py(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load mat file: {path}\n{e}") from e


def _squeeze_mat_array(a: np.ndarray) -> np.ndarray:
    # matlab often stores as (H,W,B) or (1,H,W,B) etc.
    a = np.array(a)
    a = np.squeeze(a)
    return a


def auto_detect_keys(
    mat: Dict[str, Any],
    require_cube: bool = True,
    require_gt: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    cube_key = None
    gt_key = None

    # choose cube: numeric ndarray with ndim==3 and largest size
    if require_cube:
        cand = []
        for k, v in mat.items():
            if not isinstance(v, np.ndarray):
                continue
            a = _squeeze_mat_array(v)
            if _is_numeric_ndarray(a) and a.ndim == 3:
                cand.append((a.size, k))
        if cand:
            cand.sort(reverse=True)
            cube_key = cand[0][1]

    # choose gt: numeric ndarray with ndim==2 and looks like labels (small integer range)
    if require_gt:
        cand = []
        for k, v in mat.items():
            if not isinstance(v, np.ndarray):
                continue
            a = _squeeze_mat_array(v)
            if _is_numeric_ndarray(a) and a.ndim == 2:
                aa = a
                # heuristic: label maps usually have small max (<= 1000) and many repeats
                mx = float(np.max(aa))
                mn = float(np.min(aa))
                if mx <= 5000 and mn >= 0:
                    cand.append((aa.size, k))
        if cand:
            cand.sort(reverse=True)
            gt_key = cand[0][1]

    return cube_key, gt_key


def _standardize_gt(gt: np.ndarray) -> np.ndarray:
    gt = _squeeze_mat_array(gt)
    # sometimes stored as float; cast to int64
    if gt.ndim == 3 and 1 in gt.shape:
        gt = np.squeeze(gt)
    if gt.ndim != 2:
        raise ValueError(f"GT must be 2D after squeeze, got shape={gt.shape}")
    return gt.astype(np.int64)


def _standardize_cube(cube: np.ndarray, gt_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    cube = _squeeze_mat_array(cube)

    # Some datasets are stored as (Bands, H*W) or (H*W, Bands)
    if cube.ndim == 2 and gt_shape is not None:
        H, W = gt_shape
        HW = H * W
        a, b = cube.shape
        # case: (B, HW)
        if b == HW and a <= 512:
            cube = cube.reshape(a, H, W)  # (B,H,W)
        # case: (HW, B)
        elif a == HW and b <= 512:
            cube = cube.reshape(H, W, b)  # (H,W,B)

    if cube.ndim != 3:
        raise ValueError(f"Cube must be 3D after reshape/squeeze, got shape={cube.shape}")

    # Ensure cube is (H,W,B)
    # Only move axis if one dim is clearly the spectral bands (much smaller than the other two)
    shp = list(cube.shape)
    s = sorted(shp)
    # heuristic: smallest dim is bands if it is <= 0.6 * second_smallest and <= 256
    if s[0] <= 0.6 * s[1] and s[0] <= 256:
        band_dim = int(np.argmin(shp))
        if band_dim != 2:
            cube = np.moveaxis(cube, band_dim, -1)  # put bands to last

    return cube.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--data_root", default="data")
    args = ap.parse_args()

    dcfg = load_yaml(args.dataset_cfg)
    name = str(dcfg.get("dataset", "dataset")).strip()
    raw_dir = Path(str(dcfg.get("raw_dir", Path(args.data_root) / "raw" / name)))
    cube_file = raw_dir / str(dcfg.get("cube_file", "")).strip()
    gt_file = raw_dir / str(dcfg.get("gt_file", "")).strip()

    cube_key = str(dcfg.get("cube_key", "") or "").strip()
    gt_key = str(dcfg.get("gt_key", "") or "").strip()

    if not cube_file.exists():
        raise SystemExit(f"[ERROR] cube_file not found: {cube_file}")
    if not gt_file.exists():
        raise SystemExit(f"[ERROR] gt_file not found: {gt_file}")

    print(f"[info] dataset     = {name}")
    print(f"[info] raw_dir     = {raw_dir}")
    print(f"[info] cube_file   = {cube_file.name}")
    print(f"[info] gt_file     = {gt_file.name}")

    cube_mat = load_mat_any(cube_file)
    gt_mat = load_mat_any(gt_file)

    if not cube_key or not gt_key:
        # auto-detect independently (safer)
        auto_cube_key, _ = auto_detect_keys(cube_mat, require_cube=True, require_gt=False)
        _, auto_gt_key = auto_detect_keys(gt_mat, require_cube=False, require_gt=True)
        cube_key = cube_key or (auto_cube_key or "")
        gt_key = gt_key or (auto_gt_key or "")

    if not cube_key or cube_key not in cube_mat:
        raise SystemExit(f"[ERROR] cube_key not found. cube_key='{cube_key}'. Available keys: {list(cube_mat.keys())}")
    if not gt_key or gt_key not in gt_mat:
        raise SystemExit(f"[ERROR] gt_key not found. gt_key='{gt_key}'. Available keys: {list(gt_mat.keys())}")

    cube_raw = cube_mat[cube_key]
    gt_raw = gt_mat[gt_key]

    gt = _standardize_gt(gt_raw)
    cube = _standardize_cube(cube_raw, gt_shape=gt.shape)

    if cube.shape[0] != gt.shape[0] or cube.shape[1] != gt.shape[1]:
        raise SystemExit(
            f"[ERROR] cube/gt spatial shape mismatch after standardization: cube={cube.shape} gt={gt.shape}. "
            f"Check MATLAB layout (C/F order) and dataset keys in cfg."
        )


    print(f"[info] picked cube_key={cube_key} -> shape={tuple(np.shape(cube_raw))} -> saved shape={cube.shape}")
    print(f"[info] picked gt_key  ={gt_key} -> shape={tuple(np.shape(gt_raw))} -> saved shape={gt.shape}")
    print(f"[info] gt min/max={int(gt.min())}/{int(gt.max())} unique={len(np.unique(gt))}")

    out_dir = ensure_dir(Path(args.data_root) / "processed" / name / "raw")
    np.save(Path(out_dir) / "cube.npy", cube.astype(np.float32))
    np.save(Path(out_dir) / "gt.npy", gt.astype(np.int64))
    print(f"[DONE] wrote: {out_dir}/cube.npy and {out_dir}/gt.npy")


if __name__ == "__main__":
    main()
