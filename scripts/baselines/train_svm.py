#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(8):
        if (p / "configs").exists() and ((p / "hsi3d").exists() or (p / "src").exists()):
            return p
        p = p.parent
    return Path(__file__).resolve().parents[2]


repo_root = _repo_root()
sys.path.insert(0, str(repo_root))

from hsi3d.utils.io import load_yaml, load_json, save_json, ensure_dir  # noqa: E402
from hsi3d.utils.seed import set_global_seed  # noqa: E402


def _sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_dataset_name(dataset_cfg: Dict[str, Any], cfg_path: str) -> str:
    name = (dataset_cfg.get("dataset") or dataset_cfg.get("name") or "").strip()
    if name:
        return name
    return Path(cfg_path).stem


def _resolve_baseline_cfg(dataset_name: str) -> Path:
    mapping = {"pavia_university": "pu", "PU": "pu", "pu": "pu"}
    key = mapping.get(dataset_name, dataset_name)
    return repo_root / "configs" / "baselines" / f"{key}.yaml"


def _get(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(".".join(keys))
        cur = cur[k]
    return cur


def _maybe_get(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_list(v: Any) -> list:
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _parse_gamma(v: Any) -> Any:
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip()
    if s in ("scale", "auto"):
        return s
    try:
        return float(s)
    except Exception:
        return s


def _check_disjoint(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    sa = set(map(int, a.reshape(-1)))
    sb = set(map(int, b.reshape(-1)))
    inter = sa.intersection(sb)
    if inter:
        sample = list(sorted(inter))[:10]
        raise ValueError(f"Split overlap detected between {name_a} and {name_b}: n={len(inter)} sample={sample}")


def _assert_all_labeled(indices: np.ndarray, y_all: np.ndarray, num_classes: int, split_name: str) -> None:
    y = y_all[indices]
    bad = np.where((y < 0) | (y >= int(num_classes)))[0]
    if bad.size:
        ii = indices[bad[:10]].astype(np.int64).tolist()
        yy = y[bad[:10]].astype(np.int64).tolist()
        raise ValueError(
            f"{split_name} contains unlabeled/out-of-range labels. "
            f"Expected y in [0..{int(num_classes)-1}], got samples: {list(zip(ii, yy))}. "
            f"Regenerate splits using labeled pixels only."
        )


def _cm(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    y_true = y_true.astype(np.int64, copy=False)
    y_pred = y_pred.astype(np.int64, copy=False)
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("Negative labels found in y_true/y_pred. Check splits.")
    k = y_true * int(num_classes) + y_pred
    return np.bincount(k, minlength=int(num_classes) * int(num_classes)).reshape(int(num_classes), int(num_classes))


def _metrics_from_cm(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    total = cm.sum()
    if total <= 0:
        return {"OA": 0.0, "AA": 0.0, "Kappa": 0.0}

    diag = np.diag(cm)
    oa = float(diag.sum() / total)

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.where(row_sum > 0, diag / row_sum, np.nan)
    aa = float(np.nanmean(per_class_acc))

    pe = float((row_sum * col_sum).sum() / (total * total + 1e-12))
    kappa = float((oa - pe) / (1.0 - pe + 1e-12))
    return {"OA": oa, "AA": aa, "Kappa": kappa}


def _class_counts(indices: np.ndarray, y_all: np.ndarray, num_classes: int) -> list[int]:
    y = y_all[indices]
    out = [0] * int(num_classes)
    for c in range(int(num_classes)):
        out[c] = int(np.sum(y == c))
    return out


def _cap_train_per_class(
    train_idx: np.ndarray,
    y_all: np.ndarray,
    num_classes: int,
    cap: int,
    seed: int,
) -> np.ndarray:
    if cap <= 0:
        return train_idx
    rng = np.random.default_rng(int(seed))
    y_tr = y_all[train_idx]
    chunks = []
    for c in range(int(num_classes)):
        idx_c = train_idx[y_tr == c]
        if idx_c.size == 0:
            continue
        if idx_c.size > cap:
            idx_c = rng.choice(idx_c, size=int(cap), replace=False)
        chunks.append(idx_c)
    if not chunks:
        return train_idx
    out = np.concatenate(chunks).astype(np.int64, copy=False)
    rng.shuffle(out)
    return out


def _build_pipeline(
    kernel: str,
    C: float,
    gamma: Any,
    pca: int,
    class_weight: Optional[str],
    seed: int,
    pca_whiten: bool,
    pca_svd_solver: str,
):
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.svm import SVC  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore

    steps = [("scaler", StandardScaler())]
    if int(pca) > 0:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=int(pca),
                    whiten=bool(pca_whiten),
                    svd_solver=str(pca_svd_solver),
                    random_state=int(seed),
                ),
            )
        )
    steps.append(("svm", SVC(kernel=str(kernel), C=float(C), gamma=gamma, class_weight=class_weight)))
    return Pipeline(steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)

    # CLI overrides (optional). If not provided, YAML is used.
    ap.add_argument("--kernel", choices=["rbf", "linear"], default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--gamma", default=None)
    ap.add_argument("--pca", type=int, default=None)
    ap.add_argument("--max_train_per_class", type=int, default=None)
    ap.add_argument("--class_weight", default=None)  # "balanced" or "none"

    # safety switch
    ap.add_argument("--allow_overlap", action="store_true")

    args = ap.parse_args()
    set_global_seed(args.seed)

    import sklearn  # type: ignore
    import joblib  # type: ignore

    dataset_cfg = load_yaml(args.dataset_cfg)
    dataset_name = _resolve_dataset_name(dataset_cfg, args.dataset_cfg)

    bcfg_path = _resolve_baseline_cfg(dataset_name)
    if not bcfg_path.exists():
        raise FileNotFoundError(f"SVM baseline yaml not found: {bcfg_path}")
    bcfg = load_yaml(str(bcfg_path))

    svm_cfg = _get(bcfg, ("baseline", "svm"))

    yaml_kernel = _get(svm_cfg, ("kernel",))
    yaml_C = _get(svm_cfg, ("C",))
    yaml_gamma = _get(svm_cfg, ("gamma",))
    yaml_pca = _get(svm_cfg, ("pca",))
    yaml_cap = _get(svm_cfg, ("max_train_per_class",))
    yaml_cw = _maybe_get(svm_cfg, ("class_weight",), None)

    yaml_pca_whiten = bool(_maybe_get(svm_cfg, ("pca_whiten",), False))
    yaml_pca_solver = str(_maybe_get(svm_cfg, ("pca_svd_solver",), "auto"))

    # YAML-first, CLI overrides only if explicitly passed
    kernel_v = args.kernel if args.kernel is not None else yaml_kernel
    C_v = args.C if args.C is not None else yaml_C
    gamma_v = _parse_gamma(args.gamma) if args.gamma is not None else _parse_gamma(yaml_gamma)
    pca_v = args.pca if args.pca is not None else yaml_pca
    cap = int(args.max_train_per_class) if args.max_train_per_class is not None else int(yaml_cap)

    cw_raw = args.class_weight if args.class_weight is not None else yaml_cw
    if cw_raw is None or str(cw_raw).lower() in ("none", "null", "false", "0", ""):
        class_weight = None
    elif str(cw_raw).lower() == "balanced":
        class_weight = "balanced"
    else:
        raise ValueError(f"Unsupported class_weight={cw_raw}. Use 'balanced' or null/none.")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    meta_dir = out_dir / "meta"
    ensure_dir(str(out_dir))
    ensure_dir(str(ckpt_dir))
    ensure_dir(str(meta_dir))

    split_path = Path(args.split_json)
    split = load_json(args.split_json)

    label_offset = int(dataset_cfg.get("label_offset", 1))
    num_classes = int(dataset_cfg.get("num_classes", 0) or dataset_cfg.get("n_classes", 0) or 0)
    if num_classes <= 0:
        raise ValueError("num_classes not found in dataset_cfg")

    data_root = dataset_cfg.get("data_root", "data")
    processed_raw = repo_root / data_root / "processed" / dataset_name / "raw"
    cube = np.load(processed_raw / "cube.npy").astype(np.float32, copy=False)  # (H,W,B)
    gt = np.load(processed_raw / "gt.npy").astype(np.int64, copy=False)  # (H,W)

    H, W, B = cube.shape
    X_all = cube.reshape(-1, B)
    y_all = gt.reshape(-1).astype(np.int64) - label_offset  # unlabeled -> -1

    train_idx = np.asarray(split["train_indices"], dtype=np.int64).reshape(-1)
    val_idx = np.asarray(split["val_indices"], dtype=np.int64).reshape(-1)
    test_idx = np.asarray(split["test_indices"], dtype=np.int64).reshape(-1)

    if not args.allow_overlap:
        _check_disjoint(train_idx, val_idx, "train", "val")
        _check_disjoint(train_idx, test_idx, "train", "test")
        _check_disjoint(val_idx, test_idx, "val", "test")

    # Critical: ensure splits contain labeled pixels only
    _assert_all_labeled(train_idx, y_all, num_classes, "train")
    _assert_all_labeled(val_idx, y_all, num_classes, "val")
    _assert_all_labeled(test_idx, y_all, num_classes, "test")

    counts_train_in_split = _class_counts(train_idx, y_all, num_classes)
    train_idx_used = _cap_train_per_class(train_idx, y_all, num_classes, cap=cap, seed=int(args.seed))
    counts_train_used = _class_counts(train_idx_used, y_all, num_classes)

    X_tr, y_tr = X_all[train_idx_used], y_all[train_idx_used]
    X_va, y_va = X_all[val_idx], y_all[val_idx]
    X_te, y_te = X_all[test_idx], y_all[test_idx]

    if np.unique(y_tr).size < 2:
        raise ValueError("Training split has <2 classes. SVM cannot be trained.")

    kernels = [str(x) for x in _as_list(kernel_v)]
    Cs = [float(x) for x in _as_list(C_v)]
    gammas = [_parse_gamma(x) for x in _as_list(gamma_v)]
    pcas = [int(x) for x in _as_list(pca_v)]

    if any(pp > B for pp in pcas if pp > 0):
        raise ValueError(f"pca n_components exceeds bands: B={B}, pca={pcas}")

    do_search = (len(kernels) > 1) or (len(Cs) > 1) or (len(gammas) > 1) or (len(pcas) > 1)

    best = None
    search_log = []

    if do_search:
        for ker, Cc, gg, pp in itertools.product(kernels, Cs, gammas, pcas):
            clf_try = _build_pipeline(
                kernel=ker,
                C=Cc,
                gamma=gg,
                pca=pp,
                class_weight=class_weight,
                seed=int(args.seed),
                pca_whiten=yaml_pca_whiten,
                pca_svd_solver=yaml_pca_solver,
            )
            clf_try.fit(X_tr, y_tr)
            yva_pred_try = clf_try.predict(X_va)
            cm_va_try = _cm(y_va, yva_pred_try, num_classes)
            val_m_try = _metrics_from_cm(cm_va_try)

            rec = {"kernel": ker, "C": float(Cc), "gamma": gg, "pca": int(pp), "val": val_m_try}
            search_log.append(rec)

            score = float(val_m_try["Kappa"])
            key = (score, float(val_m_try["OA"]), float(val_m_try["AA"]), -int(pp), -float(Cc))
            if best is None or key > best["key"]:
                best = {"key": key, "params": (ker, float(Cc), gg, int(pp)), "val": val_m_try}

        kernel, C, gamma, pca = best["params"]
        save_json(meta_dir / "svm_search.json", {"candidates": search_log, "best": best})
    else:
        kernel, C, gamma, pca = kernels[0], Cs[0], gammas[0], pcas[0]

    clf = _build_pipeline(
        kernel=kernel,
        C=C,
        gamma=gamma,
        pca=pca,
        class_weight=class_weight,
        seed=int(args.seed),
        pca_whiten=yaml_pca_whiten,
        pca_svd_solver=yaml_pca_solver,
    )
    clf.fit(X_tr, y_tr)

    yva_pred = clf.predict(X_va)
    yte_pred = clf.predict(X_te)

    cm_va = _cm(y_va, yva_pred, num_classes)
    cm_te = _cm(y_te, yte_pred, num_classes)

    val_m = _metrics_from_cm(cm_va)
    test_m = _metrics_from_cm(cm_te)

    model_path = ckpt_dir / "model.joblib"
    joblib.dump(clf, model_path)

    n_feat_out = int(pca) if int(pca) > 0 else int(B)

    meta = {
        "dataset": dataset_name,
        "baseline": "svm",
        "split_json": str(split_path),
        "split_sha1": _sha1_file(split_path) if split_path.exists() else None,
        "baseline_cfg": str(bcfg_path),
        "baseline_cfg_sha1": _sha1_file(bcfg_path),
        "label_offset": int(label_offset),
        "num_classes": int(num_classes),
        "n_features_in": int(B),
        "n_features_out": int(n_feat_out),
        "train_total_in_split": int(train_idx.size),
        "train_total_used": int(train_idx_used.size),
        "train_counts_in_split": counts_train_in_split,
        "train_counts_used": counts_train_used,
        "svm": {
            "kernel": str(kernel),
            "C": float(C),
            "gamma": gamma,
            "pca": int(pca),
            "pca_whiten": bool(yaml_pca_whiten),
            "pca_svd_solver": str(yaml_pca_solver),
            "max_train_per_class": int(cap),
            "class_weight": class_weight,
        },
        "search": {
            "enabled": bool(do_search),
            "n_candidates": int(len(search_log)),
            "best_val": best["val"] if best is not None else None,
        },
        "env": {
            "python": sys.version.replace("\n", " "),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "platform": os.name,
        },
    }

    metrics = {"VAL": val_m, "TEST": test_m, "meta": meta}
    save_json(out_dir / "metrics.json", metrics)

    sha10 = meta["baseline_cfg_sha1"][:10] if meta.get("baseline_cfg_sha1") else "na"
    print(
        f"[svm] cfg={bcfg_path} sha1={sha10} "
        f"kernel={kernel} C={C} gamma={gamma} pca={pca} cap={cap} cw={class_weight}"
    )
    if do_search:
        print(f"[svm-search] candidates={len(search_log)} best_val={meta['search']['best_val']}")
    print(
        f"[done] out_dir={out_dir} "
        f"val_OA={val_m['OA']:.6f} val_Kappa={val_m['Kappa']:.6f} "
        f"test_OA={test_m['OA']:.6f} test_Kappa={test_m['Kappa']:.6f}"
    )


if __name__ == "__main__":
    main()
