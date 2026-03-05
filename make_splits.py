#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from utils.io import load_yaml, ensure_dir, save_json

# Optional deps for .mat reading (Houston OUC fixed split)
try:
    import h5py  # MATLAB v7.3 (HDF5)
except Exception:
    h5py = None

try:
    from scipy.io import loadmat  # MATLAB <= v7.2
except Exception:
    loadmat = None


def _parse_seeds(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return [0]
    # formats: "0-9" or "0,1,2" or "0"
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        step = 1 if b >= a else -1
        return list(range(a, b + step, step))
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def _base_keeps(nc: int) -> Tuple[int, int, int]:
    """
    Stability floors (reviewer-safe) to avoid AA instability on tiny classes.
    Piecewise floors:
      nc >= 20: (10,5,5)
      nc >= 12: (5,3,3)
      nc >=  6: (2,2,2)
      else:    (1,1, max(0,nc-2))
    """
    nc = int(nc)
    if nc >= 20:
        return 10, 5, 5
    if nc >= 12:
        return 5, 3, 3
    if nc >= 6:
        return 2, 2, 2
    kt = 1 if nc >= 1 else 0
    kv = 1 if nc >= 2 else 0
    ktest = max(0, nc - kt - kv)
    return kt, kv, ktest


def _fit_keeps(
    nc: int,
    kt: int,
    kv: int,
    ktest: int,
    lb_tr: int,
    lb_val: int,
    lb_test: int,
    relax_order: Tuple[str, str, str] = ("train", "val", "test"),
) -> Tuple[int, int, int]:
    """
    Fit (kt,kv,ktest) into nc, while trying NOT to go below lower bounds (lb_*).
    If even sum(lb_*) > nc, then we allow going below lb_* as a last resort.
    """
    nc = int(nc)
    kt = max(0, int(kt))
    kv = max(0, int(kv))
    ktest = max(0, int(ktest))

    lb_tr = max(0, int(lb_tr))
    lb_val = max(0, int(lb_val))
    lb_test = max(0, int(lb_test))

    kt = min(kt, nc)
    kv = min(kv, nc)
    ktest = min(ktest, nc)

    kt = max(kt, lb_tr)
    kv = max(kv, lb_val)
    ktest = max(ktest, lb_test)

    def _sum() -> int:
        return int(kt + kv + ktest)

    # Phase A: reduce down to lower bounds
    excess = _sum() - nc
    if excess > 0:
        for key in relax_order:
            if excess <= 0:
                break
            if key == "train":
                reducible = max(0, kt - lb_tr)
                dec = min(excess, reducible)
                kt -= dec
                excess -= dec
            elif key == "val":
                reducible = max(0, kv - lb_val)
                dec = min(excess, reducible)
                kv -= dec
                excess -= dec
            elif key == "test":
                reducible = max(0, ktest - lb_test)
                dec = min(excess, reducible)
                ktest -= dec
                excess -= dec

    # Phase B: if still infeasible, allow going below lower bounds
    excess = _sum() - nc
    if excess > 0:
        for key in relax_order:
            if excess <= 0:
                break
            if key == "train":
                reducible = kt
                dec = min(excess, reducible)
                kt -= dec
                excess -= dec
            elif key == "val":
                reducible = kv
                dec = min(excess, reducible)
                kv -= dec
                excess -= dec
            elif key == "test":
                reducible = ktest
                dec = min(excess, reducible)
                ktest -= dec
                excess -= dec

    kt = max(0, min(int(kt), nc))
    kv = max(0, min(int(kv), nc - kt))
    ktest = max(0, min(int(ktest), nc - kt - kv))
    return int(kt), int(kv), int(ktest)


def _load_mat_any(path: str) -> Dict[str, np.ndarray]:
    """
    Load .mat (v7.3 via h5py; older via scipy.io.loadmat).
    Return dict name->ndarray for all datasets/variables.
    """
    p = str(path)

    # Try HDF5 first (MATLAB v7.3)
    if h5py is not None:
        try:
            out: Dict[str, np.ndarray] = {}
            with h5py.File(p, "r") as f:
                def walk(name, obj):
                    import h5py as _h5py
                    if isinstance(obj, _h5py.Dataset):
                        out[name] = np.asarray(obj)
                f.visititems(walk)
            if out:
                return out
        except Exception:
            pass

    # Fallback: scipy loadmat
    if loadmat is None:
        raise RuntimeError("Cannot load .mat: neither h5py (v7.3) nor scipy.io.loadmat is available.")
    m = loadmat(p)
    out = {k: np.asarray(v) for k, v in m.items() if not k.startswith("__")}
    if not out:
        raise RuntimeError(f"No arrays found in mat: {p}")
    return out


def _pick_key_from_index(keys: List[str], kind: str) -> str:
    """
    Auto pick train/test key from index.mat keys.
    kind: 'train' or 'test'
    """
    kind = kind.lower()

    def score(k: str) -> int:
        kl = k.lower()
        s = 0
        if kind == "train":
            if "train" in kl:
                s += 120
            if kl in ("tr", "train", "train_index", "train_idx", "trainind", "trainindices"):
                s += 60
        else:
            if "test" in kl:
                s += 120
            if kl in ("te", "test", "test_index", "test_idx", "testind", "testindices"):
                s += 60

        if "index" in kl or "idx" in kl or "ind" in kl:
            s += 15
        if "gt" in kl or "all" in kl:
            s -= 40
        return s

    scored = sorted([(score(k), k) for k in keys], reverse=True)
    best_score, best_key = scored[0]
    if best_score <= 0:
        raise RuntimeError(f"Failed to auto-detect {kind} key. Available keys: {keys}")
    return best_key


def _to_linear_indices(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Convert possible representations to 0-based linear indices in [0, H*W).
    Supported:
      - mask/label image of shape (H,W): arr>0 -> indices
      - linear indices: shape (N,) or (1,N) or (N,1)
      - row/col pairs: shape (2,N) or (N,2)
    Handles 1-based (MATLAB) or 0-based automatically.
    """
    a = np.asarray(arr).squeeze()

    # mask/label image
    if a.ndim == 2 and a.shape == (H, W):
        idx = np.flatnonzero(a.reshape(-1) > 0)
        return idx.astype(np.int64)

    # row/col pairs
    if a.ndim == 2 and (a.shape[0] == 2 or a.shape[1] == 2):
        rc = a if a.shape[0] == 2 else a.T
        r = np.asarray(rc[0]).reshape(-1)
        c = np.asarray(rc[1]).reshape(-1)

        m = np.isfinite(r) & np.isfinite(c)
        r = r[m].astype(np.int64)
        c = c[m].astype(np.int64)

        if r.size > 0:
            rmin, rmax = int(r.min()), int(r.max())
            cmin, cmax = int(c.min()), int(c.max())
            # common MATLAB export: 1-based indices
            one_based = (rmin >= 1 and cmin >= 1 and rmax <= H and cmax <= W)
            # some datasets (notably Houston2018 DFC) provide indices in a 2x grid (e.g., 0.5m GT vs 1m HSI)
            one_based_2x = (not one_based) and (rmin >= 1 and cmin >= 1 and rmax <= 2 * H and cmax <= 2 * W)
        else:
            # empty -> treat as one-based so we don't produce negatives
            one_based = True
            one_based_2x = False

        if one_based or one_based_2x:
            r = r - 1
            c = c - 1
            if r.size > 0:
                rmin, rmax = int(r.min()), int(r.max())
                cmin, cmax = int(c.min()), int(c.max())

        # If indices exceed the target grid but fit a 2x grid, downsample by 2.
        # This makes 0.5m (2H x 2W) coordinate lists usable with 1m (H x W) images/GT.
        if r.size > 0:
            fits_2x_zero_based = (rmax >= H or cmax >= W) and (rmax <= 2 * H - 1) and (cmax <= 2 * W - 1)
            if fits_2x_zero_based:
                import sys as _sys
                print(
                    f"[warn] rc_pairs index looks like 2x grid ({2*H}x{2*W}) while target is ({H}x{W}); "
                    f"downsampling (row//2, col//2).",
                    file=_sys.stderr,
                )
                r = r // 2
                c = c // 2

        idx = r * W + c
        idx = idx[(idx >= 0) & (idx < H * W)]
        return np.unique(idx).astype(np.int64)

    # linear index list
    a = a.reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.zeros((0,), dtype=np.int64)

    a = a.astype(np.int64)
    N = H * W
    amin, amax = int(a.min()), int(a.max())

    # 1-based if within [1, N]
    if amin >= 1 and amax <= N:
        a = a - 1

    a = a[(a >= 0) & (a < N)]
    return np.unique(a).astype(np.int64)


def _make_fixed_index_split(
    gt_flat: np.ndarray,
    classes: List[int],
    index_mat_path: str,
    seed: int,
    train_key: str = "",
    test_key: str = "",
    fixed_val_per_class: int = -1,
    val_ratio: float = 0.1,
    min_train_keep_per_class: int = 1,
    min_val_per_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Houston OUC fixed split:
      - train/test are loaded from index_mat
      - val is sampled from TRAIN only (stratified by class), controlled by:
            if fixed_val_per_class >= 0: fixed number per class
            else: use val_ratio in each class, with min_val_per_class
      - keep at least min_train_keep_per_class in TRAIN per class after taking val (when feasible)
    """
    rng = np.random.default_rng(int(seed))
    N = int(gt_flat.size)
    # infer H,W from flat length by caller? we don't have; only need N and class positions
    # But index conversion needs H,W; here we can't. So caller provides linear indices already? No.
    # => We'll load index_mat and detect H,W by requiring gt comes from 2D originally in main.
    raise RuntimeError("Internal error: _make_fixed_index_split must be called with H,W-aware wrapper.")


def _make_stratified_split(
    gt_flat: np.ndarray,
    classes: List[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_train_per_class: int = 1,
    min_val_per_class: int = 1,
    min_test_per_class: int = 1,
    per_class_train: int = -1,
    per_class_val: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Stratified RANDOM split on labeled pixels (gt>0), producing:
      train_indices, val_indices, test_indices (flat indices into H*W)

    Two modes:
      (A) ratio mode: use train_ratio/val_ratio and then enforce strict global targets
      (B) per-class mode: use fixed per_class_train/per_class_val for each class; NO strict global target adjust

    Per-class minima (min_*_per_class) are enforced as "keeps" when feasible; relaxed only for tiny classes.
    """
    rng = np.random.default_rng(int(seed))
    N = int((gt_flat > 0).sum())

    fixed_mode = (int(per_class_train) >= 0 or int(per_class_val) >= 0)
    if fixed_mode:
        if int(per_class_train) < 0 or int(per_class_val) < 0:
            raise SystemExit("[ERROR] per-class mode requires BOTH --per_class_train and --per_class_val (>=0).")

    # per-class pools
    per_class_all: Dict[int, np.ndarray] = {}
    for c in classes:
        idx_c = np.where(gt_flat == c)[0].astype(np.int64)
        rng.shuffle(idx_c)
        per_class_all[c] = idx_c

    tr_list, va_list, te_list = [], [], []
    per_class_policy: Dict[int, str] = {}

    keep_tr_map: Dict[int, int] = {}
    keep_va_map: Dict[int, int] = {}
    keep_te_map: Dict[int, int] = {}

    # For stats target
    tgt_tr = tgt_va = tgt_te = 0
    if fixed_mode:
        # compute "intended target" as sum(min(fixed, nc))
        for c in classes:
            nc = int(per_class_all[c].size)
            if nc <= 0:
                continue
            tr_t = min(int(per_class_train), nc)
            va_t = min(int(per_class_val), max(0, nc - tr_t))
            te_t = max(0, nc - tr_t - va_t)
            tgt_tr += tr_t
            tgt_va += va_t
            tgt_te += te_t
    else:
        # strict global targets in ratio mode
        tgt_tr = int(round(N * float(train_ratio)))
        tgt_va = int(round(N * float(val_ratio)))
        tgt_tr = max(0, min(tgt_tr, N))
        tgt_va = max(0, min(tgt_va, N - tgt_tr))
        tgt_te = N - tgt_tr - tgt_va

    for c in classes:
        idx_c = per_class_all[c]
        nc = int(idx_c.size)
        if nc <= 0:
            per_class_policy[c] = "empty"
            keep_tr_map[c] = keep_va_map[c] = keep_te_map[c] = 0
            continue

        # stability floors
        base_tr, base_va, base_te = _base_keeps(nc)

        req_tr = max(0, int(min_train_per_class))
        req_va = max(0, int(min_val_per_class))
        req_te = max(0, int(min_test_per_class))

        # For "keeps": in fixed_mode, also consider fixed targets as desired sizes (but they can be relaxed if nc too small)
        if fixed_mode:
            desired_tr = max(base_tr, req_tr, int(per_class_train))
            desired_va = max(base_va, req_va, int(per_class_val))
            # test desired at least req/base
            desired_te = max(base_te, req_te)
        else:
            desired_tr = max(base_tr, req_tr)
            desired_va = max(base_va, req_va)
            desired_te = max(base_te, req_te)

        keep_tr, keep_va, keep_te = _fit_keeps(
            nc=nc,
            kt=desired_tr, kv=desired_va, ktest=desired_te,
            lb_tr=base_tr, lb_val=base_va, lb_test=base_te,
            relax_order=("train", "val", "test"),
        )

        keep_tr_map[c] = int(keep_tr)
        keep_va_map[c] = int(keep_va)
        keep_te_map[c] = int(keep_te)

        relaxed = (keep_tr < desired_tr) or (keep_va < desired_va) or (keep_te < desired_te)
        per_class_policy[c] = "relaxed_min_due_to_small_class" if relaxed else "min_ok"

        if fixed_mode:
            # fixed per-class targets
            ntr = min(int(per_class_train), nc)
            nva = min(int(per_class_val), max(0, nc - ntr))

            # enforce keeps
            ntr = max(ntr, keep_tr)
            nva = max(nva, keep_va)

            # ensure room for test keep
            if ntr + nva > nc - keep_te:
                overflow = (ntr + nva) - (nc - keep_te)
                # shrink val first
                dec_va = min(overflow, max(0, nva - keep_va))
                nva -= dec_va
                overflow -= dec_va
                if overflow > 0:
                    ntr = max(keep_tr, ntr - overflow)

            # final feasibility fallback
            te_x = nc - ntr - nva
            if te_x < keep_te or ntr < keep_tr or nva < keep_va:
                # honor keeps then allocate remaining sequentially
                ntr = min(nc, keep_tr)
                nva = min(nc - ntr, keep_va)
                per_class_policy[c] = "forced_fallback_extreme"

        else:
            # ratio allocation
            ntr = int(round(nc * float(train_ratio)))
            nva = int(round(nc * float(val_ratio)))

            # apply protected minima
            ntr = max(ntr, keep_tr)
            nva = max(nva, keep_va)

            # ensure room for test keep
            if ntr + nva > nc - keep_te:
                overflow = (ntr + nva) - (nc - keep_te)
                # shrink val first but keep keep_va
                dec_va = min(overflow, max(0, nva - keep_va))
                nva -= dec_va
                overflow -= dec_va
                if overflow > 0:
                    ntr = max(keep_tr, ntr - overflow)

            # last resort
            te_x = nc - ntr - nva
            if te_x < keep_te or ntr < keep_tr or nva < keep_va:
                ntr = min(nc, keep_tr)
                nva = min(nc - ntr, keep_va)
                per_class_policy[c] = "forced_fallback_extreme"

        tr_c = idx_c[:ntr]
        va_c = idx_c[ntr:ntr + nva]
        te_c = idx_c[ntr + nva:]

        tr_list.append(tr_c)
        va_list.append(va_c)
        te_list.append(te_c)

    tr = np.concatenate(tr_list, axis=0) if tr_list else np.zeros((0,), dtype=np.int64)
    va = np.concatenate(va_list, axis=0) if va_list else np.zeros((0,), dtype=np.int64)
    te = np.concatenate(te_list, axis=0) if te_list else np.zeros((0,), dtype=np.int64)

    def _recount(arr: np.ndarray) -> Dict[int, int]:
        cnt = {c: 0 for c in classes}
        if arr.size:
            labs = gt_flat[arr]
            for cc in classes:
                cnt[cc] = int((labs == cc).sum())
        return cnt

    tr_cnt = _recount(tr)
    va_cnt = _recount(va)
    te_cnt = _recount(te)

    forced = {"train": False, "val": False}
    keep_violation = {"train": False, "val": False}

    def _move(
        src: np.ndarray,
        dst: np.ndarray,
        src_cnt: Dict[int, int],
        dst_cnt: Dict[int, int],
        need: int,
        keep_map: Dict[int, int],
        max_rounds: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        if need <= 0 or src.size == 0:
            return src, dst, 0

        moved_total = 0
        src_list = src.tolist()

        for _ in range(max_rounds):
            if moved_total >= need or len(src_list) == 0:
                break
            rng.shuffle(src_list)

            movable = []
            for idx in src_list:
                c = int(gt_flat[int(idx)])
                if src_cnt.get(c, 0) > int(keep_map.get(c, 0)):
                    movable.append(int(idx))
                    if len(movable) >= (need - moved_total):
                        break

            if not movable:
                break

            moved = np.array(movable, dtype=np.int64)
            moved_set = set(movable)
            src_list = [x for x in src_list if x not in moved_set]

            dst = moved if dst.size == 0 else np.concatenate([dst, moved], axis=0)

            for idx in moved:
                cc = int(gt_flat[int(idx)])
                src_cnt[cc] -= 1
                dst_cnt[cc] = dst_cnt.get(cc, 0) + 1

            moved_total += int(moved.size)

        src = np.array(src_list, dtype=np.int64) if src_list else np.zeros((0,), dtype=np.int64)
        return src, dst, moved_total

    # In ratio mode, adjust to strict global sizes (respect per-class keep)
    if not fixed_mode:
        if tr.size > tgt_tr:
            excess = int(tr.size - tgt_tr)
            tr, te, moved = _move(tr, te, tr_cnt, te_cnt, excess, keep_tr_map)
            if moved < excess:
                forced["train"] = True
                keep_violation["train"] = True
                rng.shuffle(tr)
                moved2 = tr[: (excess - moved)]
                tr = tr[(excess - moved):]
                te = np.concatenate([te, moved2], axis=0)
        elif tr.size < tgt_tr:
            need = int(tgt_tr - tr.size)
            te, tr, moved = _move(te, tr, te_cnt, tr_cnt, need, keep_te_map)
            if moved < need:
                forced["train"] = True
                keep_violation["train"] = True
                rng.shuffle(te)
                moved2 = te[: (need - moved)]
                te = te[(need - moved):]
                tr = np.concatenate([tr, moved2], axis=0)

        if va.size > tgt_va:
            excess = int(va.size - tgt_va)
            va, te, moved = _move(va, te, va_cnt, te_cnt, excess, keep_va_map)
            if moved < excess:
                forced["val"] = True
                keep_violation["val"] = True
                rng.shuffle(va)
                moved2 = va[: (excess - moved)]
                va = va[(excess - moved):]
                te = np.concatenate([te, moved2], axis=0)
        elif va.size < tgt_va:
            need = int(tgt_va - va.size)
            te, va, moved = _move(te, va, te_cnt, va_cnt, need, keep_te_map)
            if moved < need:
                forced["val"] = True
                keep_violation["val"] = True
                rng.shuffle(te)
                moved2 = te[: (need - moved)]
                te = te[(need - moved):]
                va = np.concatenate([va, moved2], axis=0)

    # ensure disjoint
    tr_set, va_set, te_set = set(tr.tolist()), set(va.tolist()), set(te.tolist())
    assert tr_set.isdisjoint(va_set) and tr_set.isdisjoint(te_set) and va_set.isdisjoint(te_set), "split overlap detected"

    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)

    pc_stats = {
        str(c): {
            "total": int((gt_flat == c).sum()),
            "train": int((gt_flat[tr] == c).sum()) if tr.size else 0,
            "val": int((gt_flat[va] == c).sum()) if va.size else 0,
            "test": int((gt_flat[te] == c).sum()) if te.size else 0,
            "keep_train": int(keep_tr_map.get(c, 0)),
            "keep_val": int(keep_va_map.get(c, 0)),
            "keep_test": int(keep_te_map.get(c, 0)),
            "policy": str(per_class_policy.get(c, "")),
        }
        for c in classes
    }

    stats = {
        "mode": "per_class_fixed" if fixed_mode else "ratio",
        "N_labeled": int(N),
        "target": {"train": int(tgt_tr), "val": int(tgt_va), "test": int(tgt_te)},
        "actual": {"train": int(tr.size), "val": int(va.size), "test": int(te.size)},
        "min_request": {
            "train": int(min_train_per_class),
            "val": int(min_val_per_class),
            "test": int(min_test_per_class),
        },
        "fixed_per_class": {
            "train": int(per_class_train) if fixed_mode else -1,
            "val": int(per_class_val) if fixed_mode else -1,
        },
        "forced_global_fix": forced,
        "keep_violation": keep_violation,
        "per_class": pc_stats,
    }

    return tr, va, te, stats


def _make_houston_fixed_from_index(
    gt_2d: np.ndarray,
    classes: List[int],
    index_mat_path: str,
    seed: int,
    train_key: str = "",
    test_key: str = "",
    fixed_val_per_class: int = -1,
    val_ratio: float = 0.1,
    min_train_keep_per_class: int = 1,
    min_val_per_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Wrapper with H,W for Houston fixed split.
    """
    gt = np.asarray(gt_2d).astype(np.int64)
    H, W = gt.shape
    gt_flat = gt.reshape(-1)
    N = int(gt_flat.size)

    idx_dict = _load_mat_any(index_mat_path)
    keys = list(idx_dict.keys())

    tr_key = train_key.strip() or _pick_key_from_index(keys, "train")
    te_key = test_key.strip() or _pick_key_from_index(keys, "test")

    tr_raw = idx_dict[tr_key]
    te_raw = idx_dict[te_key]

    tr_idx = np.unique(_to_linear_indices(tr_raw, H, W))
    te_idx = np.unique(_to_linear_indices(te_raw, H, W))

    # remove overlap (prefer test)
    tr_set = set(tr_idx.tolist())
    te_set = set(te_idx.tolist())
    overlap = tr_set & te_set
    if overlap:
        tr_set = tr_set - overlap
    tr_idx = np.array(sorted(tr_set), dtype=np.int64)
    te_idx = np.array(sorted(te_set), dtype=np.int64)

    # Masks for fixed train/test
    mask_tr = np.zeros(N, dtype=bool)
    mask_te = np.zeros(N, dtype=bool)
    mask_tr[tr_idx] = True
    mask_te[te_idx] = True

    # Sample val from TRAIN only
    rng = np.random.default_rng(int(seed))
    mask_val = np.zeros(N, dtype=bool)

    keep_tr = max(0, int(min_train_keep_per_class))
    min_va = max(0, int(min_val_per_class))

    for c in classes:
        pos = np.flatnonzero(mask_tr & (gt_flat == int(c)))
        if pos.size == 0:
            continue

        if int(fixed_val_per_class) >= 0:
            take = int(fixed_val_per_class)
        else:
            take = int(round(pos.size * float(val_ratio)))
            take = max(take, min_va)

        # keep at least keep_tr in train after taking val (when feasible)
        max_take = max(0, int(pos.size - keep_tr))
        take = min(take, max_take)
        if take <= 0:
            continue

        chosen = rng.choice(pos, size=take, replace=False)
        mask_val[chosen] = True

    mask_tr_final = mask_tr & (~mask_val)

    tr = np.flatnonzero(mask_tr_final).astype(np.int64)
    va = np.flatnonzero(mask_val).astype(np.int64)
    te = np.flatnonzero(mask_te).astype(np.int64)

    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)

    # Per-class stats
    pc_stats = {}
    for c in classes:
        c = int(c)
        pc_stats[str(c)] = {
            "total": int((gt_flat == c).sum()),
            "train": int((gt_flat[tr] == c).sum()) if tr.size else 0,
            "val": int((gt_flat[va] == c).sum()) if va.size else 0,
            "test": int((gt_flat[te] == c).sum()) if te.size else 0,
            "keep_train": int(keep_tr),
            "keep_val": int(min_va),
            "keep_test": 0,
            "policy": "fixed_train_test__val_from_train",
        }

    stats = {
        "mode": "fixed_index",
        "N_labeled": int((gt_flat > 0).sum()),
        "target": {"train": -1, "val": -1, "test": -1},
        "actual": {"train": int(tr.size), "val": int(va.size), "test": int(te.size)},
        "min_request": {
            "train": int(keep_tr),
            "val": int(min_va),
            "test": 0,
        },
        "fixed_index": {
            "index_mat": str(index_mat_path),
            "train_key": str(tr_key),
            "test_key": str(te_key),
            "fixed_val_per_class": int(fixed_val_per_class),
            "val_ratio_if_no_fixed": float(val_ratio),
            "val_from_train_only": True,
            "note": "train/test are fixed by index_mat; val is sampled from train only."
        },
        "forced_global_fix": {"train": False, "val": False},
        "keep_violation": {"train": False, "val": False},
        "per_class": pc_stats,
    }

    return tr, va, te, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--split_tag", default="random")
    ap.add_argument("--seeds", default="0", help='e.g. "0-9" or "0,1,2" or "0"')
    ap.add_argument("--train_ratio", type=float, default=0.10)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_dir", default="splits")

    ap.add_argument("--min_train_per_class", type=int, default=1)
    ap.add_argument("--min_val_per_class", type=int, default=1)
    ap.add_argument("--min_test_per_class", type=int, default=1)

    # fixed per-class protocol (PU / Houston2018 random stratified)
    ap.add_argument("--per_class_train", type=int, default=-1, help=">=0 enables per-class fixed protocol (train per class). Requires --per_class_val too.")
    ap.add_argument("--per_class_val", type=int, default=-1, help=">=0 enables per-class fixed protocol (val per class). Requires --per_class_train too.")

    # NEW: Houston OUC fixed train/test from index.mat
    ap.add_argument("--fixed_index", action="store_true",
                    help="Use fixed train/test from a dataset-provided index .mat (e.g., Houston OUC houston_index.mat).")
    ap.add_argument("--fixed_index_mat", default="",
                    help="Path to index .mat. If empty, will try data/raw/<dataset>/houston_index.mat.")
    ap.add_argument("--fixed_train_key", default="",
                    help="Optional override: train key name inside index .mat (e.g., TR / train_index / train).")
    ap.add_argument("--fixed_test_key", default="",
                    help="Optional override: test key name inside index .mat (e.g., TE / test_index / test).")
    ap.add_argument("--fixed_val_per_class", type=int, default=-1,
                    help="If >=0, sample this many val pixels per class FROM TRAIN ONLY (fixed split mode). If <0, use --val_ratio + --min_val_per_class.")

    ap.add_argument("--stats_csv_name", default="", help="optional override csv filename (default: <dataset>_perclass_seeds.csv)")
    args = ap.parse_args()

    dcfg = load_yaml(args.dataset_cfg)
    name = str(dcfg.get("dataset", "dataset"))
    label_offset = int(dcfg.get("label_offset", 1))

    proc = Path(args.data_root) / "processed" / name / "raw"
    gt_p = proc / "gt.npy"
    if not gt_p.exists():
        raise SystemExit(f"[ERROR] processed gt not found: {gt_p}. Run prepare_raw_to_processed.py first.")
    gt = np.load(gt_p).astype(np.int64)
    if gt.ndim != 2:
        raise SystemExit(f"[ERROR] processed gt.npy must be 2D (H,W). got {gt.shape}")
    gt_flat = gt.reshape(-1)

    classes = sorted([int(c) for c in np.unique(gt_flat) if int(c) > 0])
    if not classes:
        raise SystemExit("[ERROR] no labeled pixels (gt>0) found.")

    out_base = ensure_dir(Path(args.out_dir) / str(args.split_tag))
    seeds = _parse_seeds(args.seeds)

    perclass_rows: List[Dict[str, Any]] = []

    # auto fixed_index_mat if needed
    fixed_index_mat = str(args.fixed_index_mat).strip()
    if args.fixed_index and not fixed_index_mat:
        cand = Path(args.data_root) / "raw" / name / "houston_index.mat"
        if cand.exists():
            fixed_index_mat = str(cand)
        else:
            raise SystemExit("[ERROR] --fixed_index is set but --fixed_index_mat is empty and default cand not found: "
                             f"{cand}")

    for sd in seeds:
        if args.fixed_index:
            tr, va, te, stats = _make_houston_fixed_from_index(
                gt_2d=gt,
                classes=classes,
                index_mat_path=fixed_index_mat,
                seed=int(sd),
                train_key=str(args.fixed_train_key),
                test_key=str(args.fixed_test_key),
                fixed_val_per_class=int(args.fixed_val_per_class),
                val_ratio=float(args.val_ratio),
                min_train_keep_per_class=int(args.min_train_per_class),
                min_val_per_class=int(args.min_val_per_class),
            )
        else:
            tr, va, te, stats = _make_stratified_split(
                gt_flat=gt_flat,
                classes=classes,
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                seed=int(sd),
                min_train_per_class=int(args.min_train_per_class),
                min_val_per_class=int(args.min_val_per_class),
                min_test_per_class=int(args.min_test_per_class),
                per_class_train=int(args.per_class_train),
                per_class_val=int(args.per_class_val),
            )

        out = {
            "dataset": name,
            "split_tag": str(args.split_tag),
            "seed": int(sd),
            "label_offset": int(label_offset),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "per_class_train": int(args.per_class_train),
            "per_class_val": int(args.per_class_val),
            "fixed_index": bool(args.fixed_index),
            "fixed_index_mat": str(fixed_index_mat) if args.fixed_index else "",
            "fixed_val_per_class": int(args.fixed_val_per_class) if args.fixed_index else -1,
            "train_indices": tr.astype(np.int64).tolist(),
            "val_indices": va.astype(np.int64).tolist(),
            "test_indices": te.astype(np.int64).tolist(),
            "stats": stats,
        }

        out_path = Path(out_base) / f"{name}_seed{sd}.json"
        save_json(str(out_path), out)
        print(f"[ok] wrote {out_path} | sizes: tr={len(out['train_indices'])} va={len(out['val_indices'])} te={len(out['test_indices'])}")

        pc = (stats or {}).get("per_class", {})
        tgt = (stats or {}).get("target", {})
        act = (stats or {}).get("actual", {})
        nlab = int((stats or {}).get("N_labeled", 0))
        min_req = (stats or {}).get("min_request", {}) or {}
        forced = (stats or {}).get("forced_global_fix", {}) or {}
        keep_v = (stats or {}).get("keep_violation", {}) or {}
        mode = str((stats or {}).get("mode", ""))

        for c in classes:
            ckey = str(c)
            d = pc.get(ckey, {})

            total = int(d.get("total", 0))
            trn = int(d.get("train", 0))
            val = int(d.get("val", 0))
            tes = int(d.get("test", 0))

            perclass_rows.append({
                "dataset": name,
                "split_tag": str(args.split_tag),
                "seed": int(sd),
                "class": int(c),
                "label_offset": int(label_offset),
                "mode": mode,

                "total": total,
                "train": trn,
                "val": val,
                "test": tes,

                "train_frac_in_class": (trn / total) if total > 0 else 0.0,
                "val_frac_in_class": (val / total) if total > 0 else 0.0,
                "test_frac_in_class": (tes / total) if total > 0 else 0.0,

                "keep_train": int(d.get("keep_train", 0)),
                "keep_val": int(d.get("keep_val", 0)),
                "keep_test": int(d.get("keep_test", 0)),
                "policy": str(d.get("policy", "")),

                "N_labeled": nlab,
                "train_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "per_class_train": int(args.per_class_train),
                "per_class_val": int(args.per_class_val),

                "target_train": int(tgt.get("train", 0)),
                "target_val": int(tgt.get("val", 0)),
                "target_test": int(tgt.get("test", 0)),
                "actual_train": int(act.get("train", 0)),
                "actual_val": int(act.get("val", 0)),
                "actual_test": int(act.get("test", 0)),

                "min_train_per_class_req": int(min_req.get("train", 0)),
                "min_val_per_class_req": int(min_req.get("val", 0)),
                "min_test_per_class_req": int(min_req.get("test", 0)),

                "forced_train_fix": bool(forced.get("train", False)),
                "forced_val_fix": bool(forced.get("val", False)),
                "keep_violation_train": bool(keep_v.get("train", False)),
                "keep_violation_val": bool(keep_v.get("val", False)),

                "fixed_index": bool(args.fixed_index),
                "fixed_index_mat": str(fixed_index_mat) if args.fixed_index else "",
                "fixed_val_per_class": int(args.fixed_val_per_class) if args.fixed_index else -1,
                "fixed_train_key": str(args.fixed_train_key) if args.fixed_index else "",
                "fixed_test_key": str(args.fixed_test_key) if args.fixed_index else "",
            })

    if perclass_rows:
        csv_name = args.stats_csv_name.strip() or f"{name}_perclass_seeds.csv"
        stats_csv = Path(out_base) / csv_name
        fieldnames = list(perclass_rows[0].keys())
        with open(stats_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in perclass_rows:
                w.writerow(r)
        print(f"[ok] wrote {stats_csv} (single stats csv; per-class rows = {len(perclass_rows)})")

    print("[DONE] make_splits finished.")


if __name__ == "__main__":
    main()
