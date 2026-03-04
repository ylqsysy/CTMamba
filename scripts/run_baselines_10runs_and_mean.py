#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run baseline experiments for multiple seeds and aggregate mean/std.

Supports baselines:
- svm
- 2dcnn
- 3dcnn
- a2s2k
- spectralformer
- ssftt
- 3dss_mamba

Robust to metrics.json schema differences:
- accepts {"VAL": {...}, "TEST": {...}} or {"val": {...}, "test": {...}}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "configs").exists() and (p / "src" / "hsi3d").exists():
            return p
        p = p.parent
    # fallback: two levels up from scripts/
    return Path(__file__).resolve().parents[1]


repo_root = _repo_root()
sys.path.insert(0, str(repo_root))


from hsi3d.utils.io import load_yaml, ensure_dir, save_json  # noqa: E402


def _parse_seeds(spec: str) -> List[int]:
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def _cfg_name(dataset: str) -> str:
    return "pu" if dataset == "pavia_university" else dataset


def _baseline_cfg_path(dataset: str) -> Path:
    return repo_root / "configs" / "baselines" / f"{_cfg_name(dataset)}.yaml"


def _train_cfg_path(dataset: str) -> Path:
    return repo_root / "configs" / "train" / f"{_cfg_name(dataset)}.yaml"


def _dataset_cfg_path(dataset: str) -> Path:
    return repo_root / "configs" / "datasets" / f"{dataset}.yaml"


def _load_baseline_section(dataset: str, baseline: str) -> Dict[str, Any]:
    p = _baseline_cfg_path(dataset)
    cfg = load_yaml(str(p))
    # common layouts:
    # 1) {"baseline": {"svm": {...}, ...}}
    # 2) {"svm": {...}, ...}
    if isinstance(cfg, dict) and "baseline" in cfg and isinstance(cfg["baseline"], dict):
        sec = cfg["baseline"].get(baseline)
    else:
        sec = cfg.get(baseline) if isinstance(cfg, dict) else None
    if not isinstance(sec, dict):
        raise KeyError(f"Missing baseline section '{baseline}' in {p}")
    return sec


def _sha1_of_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_metrics_block(m: Dict[str, Any], want: str) -> Dict[str, Any]:
    # want: "VAL" or "TEST"
    cands = [want, want.lower(), want.capitalize()]
    for k in cands:
        v = m.get(k)
        if isinstance(v, dict):
            return v
    # some writers may nest under "metrics"
    mm = m.get("metrics")
    if isinstance(mm, dict):
        for k in cands:
            v = mm.get(k)
            if isinstance(v, dict):
                return v
    raise KeyError(f"metrics.json missing '{want}' block. Top keys={list(m.keys())}")


def _get_metric(d: Dict[str, Any], key: str) -> float:
    # accepts "Kappa" or "kappa"
    if key in d:
        return float(d[key])
    lk = key.lower()
    if lk in d:
        return float(d[lk])
    raise KeyError(f"Missing metric '{key}'. Keys={list(d.keys())}")


def _mean_std(rows: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    xs = np.array([r[key] for r in rows], dtype=np.float64)
    return float(xs.mean()), float(xs.std(ddof=0))


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_md(path: Path, rows: List[Dict[str, float]], mean: Dict[str, float]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    md = []
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        md.append("| " + " | ".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    md.append("")
    md.append("## Mean (10 runs)")
    md.append("```json")
    md.append(json.dumps(mean, indent=2, ensure_ascii=False))
    md.append("```")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pavia_university", "honghu", "hanchuan", "houston2013"])
    ap.add_argument(
        "--baseline",
        required=True,
        choices=["svm", "2dcnn", "3dcnn", "a2s2k", "spectralformer", "ssftt", "3dss_mamba"],
    )
    ap.add_argument("--split_tag", default="random")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out_base", default="outputs/baselines")
    ap.add_argument("--python", default=sys.executable)

    # This flag is only forwarded to baselines that support AMP.
    ap.add_argument("--use_amp", action="store_true")
    # Optional consistency check; 0 means skip
    ap.add_argument("--patch_size", type=int, default=0)
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    out_base = Path(args.out_base)
    ensure_dir(str(out_base))

    dataset_cfg = _dataset_cfg_path(args.dataset)
    train_cfg = _train_cfg_path(args.dataset)

    if not dataset_cfg.exists():
        raise FileNotFoundError(f"Missing dataset_cfg: {dataset_cfg}")
    if not train_cfg.exists():
        raise FileNotFoundError(f"Missing train_cfg: {train_cfg}")

    # baseline cfg section (for patch_size lock & optional amp check)
    bsec = _load_baseline_section(args.dataset, args.baseline)
    yaml_patch_size = int(bsec.get("patch_size", 15))

    # lock your protocol: patch_size must be 15
    if yaml_patch_size != 15:
        raise ValueError(f"[lock] baseline.{args.baseline}.patch_size must be 15 (got {yaml_patch_size})")

    if args.patch_size and args.patch_size != yaml_patch_size:
        raise ValueError(f"--patch_size={args.patch_size} does not match YAML patch_size={yaml_patch_size}")

    # cache a simple config fingerprint for debugging
    cfg_fingerprint = {
        "dataset_cfg_sha1": _sha1_of_file(dataset_cfg),
        "train_cfg_sha1": _sha1_of_file(train_cfg),
        "baseline_cfg_sha1": _sha1_of_file(_baseline_cfg_path(args.dataset)),
    }

    all_rows: List[Dict[str, float]] = []

    for seed in seeds:
        out_dir = out_base / f"{args.dataset}_{args.baseline.upper()}_seed{seed}"
        ensure_dir(str(out_dir))

        split_json = repo_root / "splits" / args.split_tag / f"{args.dataset}_seed{seed}.json"
        if not split_json.exists():
            raise FileNotFoundError(f"Missing split_json: {split_json}")

        if args.baseline == "svm":
            train_script = repo_root / "scripts" / "baselines" / "train_svm.py"
            cmd = [
                args.python, "-u", str(train_script),
                "--dataset_cfg", str(dataset_cfg),
                "--train_cfg", str(train_cfg),
                "--split_json", str(split_json),
                "--out_dir", str(out_dir),
                "--seed", str(seed),
                "--baseline", "svm",
            ]
        elif args.baseline in ("2dcnn", "3dcnn"):
            train_script = repo_root / "scripts" / "baselines" / "train_cnn.py"
            cmd = [
                args.python, "-u", str(train_script),
                "--dataset_cfg", str(dataset_cfg),
                "--train_cfg", str(train_cfg),
                "--split_json", str(split_json),
                "--out_dir", str(out_dir),
                "--seed", str(seed),
                "--baseline", args.baseline,
                "--patch_size", str(yaml_patch_size),
            ]
            if args.use_amp:
                cmd.append("--use_amp")
        elif args.baseline == "a2s2k":
            train_script = repo_root / "scripts" / "baselines" / "train_a2s2k.py"
            cmd = [
                args.python, "-u", str(train_script),
                "--dataset_cfg", str(dataset_cfg),
                "--train_cfg", str(train_cfg),
                "--split_json", str(split_json),
                "--out_dir", str(out_dir),
                "--seed", str(seed),
                "--baseline", "a2s2k",
                "--patch_size", str(yaml_patch_size),
            ]
            if args.use_amp:
                cmd.append("--use_amp")
        elif args.baseline == "spectralformer":
            train_script = repo_root / "scripts" / "baselines" / "train_spectralformer.py"
            cmd = [
                args.python, "-u", str(train_script),
                "--dataset_cfg", str(dataset_cfg),
                "--train_cfg", str(train_cfg),
                "--split_json", str(split_json),
                "--out_dir", str(out_dir),
                "--seed", str(seed),
                "--baseline", "spectralformer",
                "--patch_size", str(yaml_patch_size),
            ]
            if args.use_amp:
                cmd.append("--use_amp")
        elif args.baseline == "ssftt":
            train_script = repo_root / "scripts" / "baselines" / "train_ssftt.py"
            cmd = [
                args.python, "-u", str(train_script),
                "--dataset_cfg", str(dataset_cfg),
                "--train_cfg", str(train_cfg),
                "--split_json", str(split_json),
                "--out_dir", str(out_dir),
                "--seed", str(seed),
                "--baseline", "ssftt",
                "--patch_size", str(yaml_patch_size),
            ]
            if args.use_amp:
                cmd.append("--use_amp")
        elif args.baseline == "3dss_mamba":
            train_script = repo_root / "scripts" / "baselines" / "train_3dss_mamba.py"
            cmd = [
                args.python, "-u", str(train_script),
                "--dataset_cfg", str(dataset_cfg),
                "--train_cfg", str(train_cfg),
                "--split_json", str(split_json),
                "--out_dir", str(out_dir),
                "--seed", str(seed),
                "--baseline", "3dss_mamba",
                "--patch_size", str(yaml_patch_size),
            ]
            if args.use_amp:
                cmd.append("--use_amp")
        else:
            raise ValueError(f"Unknown baseline: {args.baseline}")

        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)

        mpath = out_dir / "metrics.json"
        if not mpath.exists():
            raise FileNotFoundError(f"Missing metrics: {mpath}")

        m = _read_json(mpath)
        v = _pick_metrics_block(m, "VAL")
        t = _pick_metrics_block(m, "TEST")

        row = {
            "seed": float(seed),
            "val_OA": _get_metric(v, "OA"),
            "val_AA": _get_metric(v, "AA"),
            "val_Kappa": _get_metric(v, "Kappa"),
            "test_OA": _get_metric(t, "OA"),
            "test_AA": _get_metric(t, "AA"),
            "test_Kappa": _get_metric(t, "Kappa"),
        }
        all_rows.append(row)

    mean: Dict[str, float] = {}
    for k in ["val_OA", "val_AA", "val_Kappa", "test_OA", "test_AA", "test_Kappa"]:
        mu, sd = _mean_std(all_rows, k)
        mean[k] = mu
        mean[k + "_std"] = sd

    out = {
        "dataset": args.dataset,
        "baseline": args.baseline,
        "split_tag": args.split_tag,
        "seeds": seeds,
        "mean": mean,
        "meta": {
            "cfg_fingerprint": cfg_fingerprint,
        },
    }

    out_mean_dir = out_base / f"{args.dataset}_{args.baseline.upper()}_mean{len(seeds)}"
    ensure_dir(str(out_mean_dir))

    save_json(str(out_mean_dir / "mean_metrics.json"), out)
    _write_csv(out_mean_dir / "summary.csv", all_rows)
    _write_md(out_mean_dir / "summary.md", all_rows, mean)

    print("[mean saved]", out_mean_dir / "mean_metrics.json")
    print(json.dumps(mean, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
