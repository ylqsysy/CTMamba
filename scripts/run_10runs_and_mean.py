#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _run_capture(cmd: List[str]) -> str:
    print("[cmd]", " ".join(cmd), flush=True)
    p = subprocess.run(
        cmd,
        check=False,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = p.stdout or ""
    if out:
        print(out, end="")
    if p.returncode != 0:
        raise SystemExit(f"[ERROR] command failed (exit={p.returncode}). See output above.")
    return out


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _sanitize_tag(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "dataset"


def _infer_paths(dataset: str) -> Tuple[Path, Path, Path]:
    d = REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml"

    model_map = {
        "pavia_university": "vssm3d_pu.yaml",
        "houston2013": "vssm3d_houston2013.yaml",
        "hanchuan": "vssm3d_hanchuan.yaml",
        "honghu": "vssm3d_honghu.yaml",
    }
    m = REPO_ROOT / "configs" / "model" / model_map[dataset]

    train_map = {
        "pavia_university": "pu.yaml",
        "houston2013": "houston2013.yaml",
        "hanchuan": "hanchuan.yaml",
        "honghu": "honghu.yaml",
    }
    t = REPO_ROOT / "configs" / "train" / train_map[dataset]
    return d, m, t


def _pick_split_dict(d: Dict[str, Any], split: str) -> Dict[str, Any]:
    """
    Accept multiple formats:
      - {"val": {...}, "test": {...}}
      - {"VAL": {...}, "TEST": {...}}
      - {"metrics": {"val": {...}, "test": {...}}}
      - {"metrics": {"VAL": {...}, "TEST": {...}}}
      - {"valid": {...}} / {"VALID": {...}} as val alias
    """
    if not isinstance(d, dict):
        raise KeyError(f"metrics root is not a dict: {type(d)}")

    def _get(container: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if k in container:
                return container[k]
        return None

    split_l = split.lower()
    split_u = split.upper()

    if split_l == "val":
        keys = ["val", "VAL", "valid", "VALID", "val_metrics", "VAL_METRICS"]
    else:
        keys = ["test", "TEST", "te", "TE", "test_metrics", "TEST_METRICS"]

    got = _get(d, keys)
    if isinstance(got, dict):
        return got

    md = d.get("metrics", None)
    if isinstance(md, dict):
        got2 = _get(md, keys)
        if isinstance(got2, dict):
            return got2

    raise KeyError(f"missing split '{split}' in metrics json. keys={list(d.keys())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=[
            "pavia_university",
            "houston2013",
            "hanchuan",
            "honghu",
        ],
    )
    ap.add_argument("--split_tag", default="random", choices=["random"])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_base", default="outputs/checkpoints")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    dataset = str(args.dataset)
    tag = _sanitize_tag(dataset)
    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    dataset_cfg, model_cfg, train_cfg = _infer_paths(dataset)
    split_dir = REPO_ROOT / "splits" / args.split_tag

    seeds: List[int] = []
    s = str(args.seeds).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in s.split(",") if x.strip()]

    print("[info] repo_root   =", REPO_ROOT)
    print("[info] dataset     =", dataset)
    print("[info] seeds       =", seeds)
    print("[info] dataset_cfg =", dataset_cfg)
    print("[info] model_cfg   =", model_cfg)
    print("[info] train_cfg   =", train_cfg)

    keys = ["OA", "AA", "Kappa"]
    per_seed = {"val": {}, "test": {}}
    test_vals = {k: [] for k in keys}
    val_vals = {k: [] for k in keys}

    for seed in seeds:
        out_dir = out_base / f"{tag}_seed{seed}"
        split_json = split_dir / f"{dataset}_seed{seed}.json"
        if not split_json.exists():
            raise SystemExit(f"[ERROR] split json not found: {split_json}")

        ckpt_best = out_dir / "checkpoints" / "best.pt"
        metrics_json = out_dir / "metrics.json"
        eval_json = out_dir / "eval.json"

        if not ckpt_best.exists():
            cmd_train = [
                str(args.python),
                "scripts/train.py",
                "--dataset_cfg", str(dataset_cfg),
                "--model_cfg", str(model_cfg),
                "--train_cfg", str(train_cfg),
                "--split_json", str(split_json),
                "--data_root", str(args.data_root),
                "--seed", str(seed),
                "--out_dir", str(out_dir),
                "--num_workers", str(int(args.num_workers)),
            ]
            if args.amp:
                cmd_train.append("--amp")


            _run(cmd_train)
        else:
            print("[info] skip train: found", ckpt_best)

        if metrics_json.exists():
            d = _load_json(metrics_json)
        elif eval_json.exists():
            d = _load_json(eval_json)
        else:
            cmd_eval = [
                str(args.python),
                "scripts/eval.py",
                "--dataset_cfg", str(dataset_cfg),
                "--model_cfg", str(model_cfg),
                "--split_json", str(split_json),
                "--data_root", str(args.data_root),
                "--seed", str(seed),
                "--ckpt", str(ckpt_best),
                "--ckpt_key", "model",
                "--batch_size", str(int(args.eval_batch_size)),
                "--num_workers", str(int(args.num_workers)),
                "--out", str(eval_json),
            ]
            if args.amp:
                cmd_eval.append("--amp")
            _run_capture(cmd_eval)
            d = _load_json(eval_json)

        v = _pick_split_dict(d, "val")
        t = _pick_split_dict(d, "test")

        per_seed["val"][str(seed)] = {k: float(v[k]) for k in keys}
        per_seed["test"][str(seed)] = {k: float(t[k]) for k in keys}
        for k in keys:
            val_vals[k].append(float(v[k]))
            test_vals[k].append(float(t[k]))

    def mean_std(xs: List[float]):
        a = np.asarray(xs, dtype=float)
        return float(a.mean()), float(a.std(ddof=0))

    mean = {"val": {}, "test": {}}
    std = {"val": {}, "test": {}}
    for k in keys:
        mean["val"][k], std["val"][k] = mean_std(val_vals[k])
        mean["test"][k], std["test"][k] = mean_std(test_vals[k])

    mean_dir = out_base / f"{tag}_mean10"
    mean_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "per_seed": per_seed,
        "mean": mean,
        "std": std,
        "meta": {
            "dataset": dataset,
            "split_tag": args.split_tag,
            "protocol": "RANDOM only; 10 seeds; STRICT split_json indices; no TTA; mean±std reported",
        },
    }
    _write_json(mean_dir / "mean_metrics.json", out)

    lines = ["seed,val_OA,val_AA,val_Kappa,test_OA,test_AA,test_Kappa"]
    for seed in seeds:
        vv = per_seed["val"][str(seed)]
        tt = per_seed["test"][str(seed)]
        lines.append(f"{seed},{vv['OA']},{vv['AA']},{vv['Kappa']},{tt['OA']},{tt['AA']},{tt['Kappa']}")
    lines.append(f"mean,{mean['val']['OA']},{mean['val']['AA']},{mean['val']['Kappa']},{mean['test']['OA']},{mean['test']['AA']},{mean['test']['Kappa']}")
    lines.append(f"std,{std['val']['OA']},{std['val']['AA']},{std['val']['Kappa']},{std['test']['OA']},{std['test']['AA']},{std['test']['Kappa']}")
    (mean_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def f4(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return ""

    md = []
    md.append(f"# {dataset} ({args.split_tag}) 10-seed summary\n")
    md.append("| seed | val_OA | val_AA | val_Kappa | test_OA | test_AA | test_Kappa |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for seed in seeds:
        vv = per_seed["val"][str(seed)]
        tt = per_seed["test"][str(seed)]
        md.append(f"| {seed} | {f4(vv['OA'])} | {f4(vv['AA'])} | {f4(vv['Kappa'])} | {f4(tt['OA'])} | {f4(tt['AA'])} | {f4(tt['Kappa'])} |")
    md.append(f"| mean | {f4(mean['val']['OA'])} | {f4(mean['val']['AA'])} | {f4(mean['val']['Kappa'])} | {f4(mean['test']['OA'])} | {f4(mean['test']['AA'])} | {f4(mean['test']['Kappa'])} |")
    md.append(f"| std  | {f4(std['val']['OA'])} | {f4(std['val']['AA'])} | {f4(std['val']['Kappa'])} | {f4(std['test']['OA'])} | {f4(std['test']['AA'])} | {f4(std['test']['Kappa'])} |")
    (mean_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("[DONE] wrote:", mean_dir / "mean_metrics.json")


if __name__ == "__main__":
    main()
