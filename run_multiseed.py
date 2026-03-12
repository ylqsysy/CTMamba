#!/usr/bin/env python3
"""Run multi-seed training, evaluation, and metric aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent


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


def _load_yaml(p: Path) -> Dict[str, Any]:
    try:
        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _signature_matches(meta: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    if not isinstance(meta, dict):
        return False
    for k, v in expected.items():
        if str(meta.get(k, "")) != str(v):
            return False
    return True


def _sanitize_tag(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "dataset"


def _infer_paths(dataset: str) -> Tuple[Path, Path, Path]:
    d = REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml"

    model_map = {
        "pavia_university": "ctmamba_pavia_university.yaml",
        "houston2013": "ctmamba_houston2013.yaml",
        "hanchuan": "ctmamba_hanchuan.yaml",
        "honghu": "ctmamba_honghu.yaml",
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


def _parse_seeds(spec: str) -> List[int]:
    s = str(spec).strip()
    if not s:
        raise SystemExit("[ERROR] empty --seeds.")
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        step = 1 if b >= a else -1
        return list(range(a, b + step, step))
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise SystemExit("[ERROR] empty --seeds list.")
    return vals


def _pick_split_dict(d: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Read split metrics from the supported JSON layouts."""
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


def _try_load_metrics(
    metrics_json: Path,
    eval_json: Path,
    *,
    expected_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any] | None, Path | None]:
    for p in (metrics_json, eval_json):
        if not p.exists():
            continue
        try:
            d = _load_json(p)
            _pick_split_dict(d, "val")
            _pick_split_dict(d, "test")
            if expected_meta is not None:
                meta = d.get("meta", {})
                if not _signature_matches(meta, expected_meta):
                    print(f"[warn] stale metrics file ignored (signature mismatch): {p}")
                    continue
            return d, p
        except Exception as e:
            print(f"[warn] invalid metrics file ignored: {p} ({e})")
    return None, None


def _mean_metrics_complete(
    mean_json: Path,
    seeds: List[int],
    *,
    expected_meta: Optional[Dict[str, Any]] = None,
) -> bool:
    if not mean_json.exists():
        return False
    try:
        d = _load_json(mean_json)
        per = d.get("per_seed", {})
        val = per.get("val", {})
        test = per.get("test", {})
        if not isinstance(val, dict) or not isinstance(test, dict):
            return False
        need = {str(int(s)) for s in seeds}
        if not need.issubset(set(val.keys())) or not need.issubset(set(test.keys())):
            return False
        for s in seeds:
            sv = val[str(int(s))]
            st = test[str(int(s))]
            for k in ("OA", "AA", "Kappa"):
                float(sv[k])
                float(st[k])
        if expected_meta is not None:
            meta = d.get("meta", {})
            if not _signature_matches(meta, expected_meta):
                return False
        return True
    except Exception:
        return False


def _resolve_runtime(
    *,
    train_cfg_obj: Dict[str, Any],
    cli_num_workers: int,
    cli_eval_batch_size: int,
    cli_eval_log_interval: int,
) -> Tuple[int, int, int]:
    def _coerce_int(v: Any, default: int, minimum: int) -> int:
        try:
            iv = int(v)
        except Exception:
            return int(default)
        return int(max(minimum, iv))

    if int(cli_num_workers) >= 0:
        num_workers = int(cli_num_workers)
    else:
        num_workers = _coerce_int(train_cfg_obj.get("num_workers", 0), default=0, minimum=0)

    if int(cli_eval_batch_size) > 0:
        eval_batch_size = int(cli_eval_batch_size)
    else:
        eval_batch_size = _coerce_int(train_cfg_obj.get("eval_batch_size", 512), default=512, minimum=1)

    if int(cli_eval_log_interval) >= 0:
        eval_log_interval = int(cli_eval_log_interval)
    else:
        eval_log_interval = _coerce_int(train_cfg_obj.get("eval_log_interval", 25), default=25, minimum=0)

    return num_workers, eval_batch_size, eval_log_interval


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
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help=">=0 to force value; -1 to read from train_cfg(num_workers).",
    )
    ap.add_argument(
        "--eval_batch_size",
        type=int,
        default=-1,
        help=">0 to force value; <=0 to read from train_cfg(eval_batch_size).",
    )
    ap.add_argument(
        "--eval_log_interval",
        type=int,
        default=-1,
        help=">=0 to force value; -1 to read from train_cfg(eval_log_interval, default=25).",
    )
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dataset_cfg_path", default="", help="Optional explicit dataset config path.")
    ap.add_argument("--model_cfg_path", default="", help="Optional explicit model config path.")
    ap.add_argument("--train_cfg_path", default="", help="Optional explicit train config path.")
    ap.add_argument("--force", action="store_true", help="Force rerun even if complete mean_metrics already exists.")
    args = ap.parse_args()

    dataset = str(args.dataset)
    tag = _sanitize_tag(dataset)
    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    def _pick_path(override: str, inferred: Path, name: str) -> Path:
        if str(override).strip():
            p = Path(str(override).strip())
        else:
            p = inferred
        if not p.exists():
            raise SystemExit(f"[ERROR] {name} not found: {p}")
        return p

    inf_dataset_cfg, inf_model_cfg, inf_train_cfg = _infer_paths(dataset)
    dataset_cfg = _pick_path(args.dataset_cfg_path, inf_dataset_cfg, "dataset_cfg")
    model_cfg = _pick_path(args.model_cfg_path, inf_model_cfg, "model_cfg")
    train_cfg = _pick_path(args.train_cfg_path, inf_train_cfg, "train_cfg")
    cfg_meta = {
        "dataset_cfg_sha1": _sha1_file(dataset_cfg),
        "model_cfg_sha1": _sha1_file(model_cfg),
        "train_cfg_sha1": _sha1_file(train_cfg),
    }
    train_cfg_obj = _load_yaml(train_cfg)
    split_dir = REPO_ROOT / "splits" / args.split_tag

    num_workers, eval_batch_size, eval_log_interval = _resolve_runtime(
        train_cfg_obj=train_cfg_obj,
        cli_num_workers=int(args.num_workers),
        cli_eval_batch_size=int(args.eval_batch_size),
        cli_eval_log_interval=int(args.eval_log_interval),
    )

    seeds = _parse_seeds(str(args.seeds))
    n_seeds = len(seeds)
    if n_seeds <= 0:
        raise SystemExit("[ERROR] no valid seeds.")

    print("[info] repo_root   =", REPO_ROOT)
    print("[info] dataset     =", dataset)
    print("[info] seeds       =", seeds)
    print("[info] dataset_cfg =", dataset_cfg)
    print("[info] model_cfg   =", model_cfg)
    print("[info] train_cfg   =", train_cfg)
    print("[info] num_workers =", num_workers)
    print("[info] eval_bs     =", eval_batch_size)
    print("[info] eval_log_it =", eval_log_interval)

    mean_dir = out_base / f"{tag}_mean{n_seeds}"
    mean_json = mean_dir / "mean_metrics.json"
    if (not bool(args.force)) and _mean_metrics_complete(mean_json, seeds, expected_meta=cfg_meta):
        print(f"[info] skip dataset: complete result exists -> {mean_json}")
        return

    keys = ["OA", "AA", "Kappa"]
    per_seed = {"val": {}, "test": {}}
    per_seed_full = {"val": {}, "test": {}, "meta": {}}
    test_vals = {k: [] for k in keys}
    val_vals = {k: [] for k in keys}
    run_time_vals: List[float] = []
    params_total_vals: List[float] = []
    params_trainable_vals: List[float] = []

    for seed in seeds:
        out_dir = out_base / f"{tag}_seed{seed}"
        split_json = split_dir / f"{dataset}_seed{seed}.json"
        if not split_json.exists():
            raise SystemExit(f"[ERROR] split json not found: {split_json}")

        ckpt_best = out_dir / "checkpoints" / "best.pt"
        metrics_json = out_dir / "metrics.json"
        eval_json = out_dir / "eval.json"
        run_sig_json = out_dir / "meta" / "run_signature.json"
        split_json_sha1 = _sha1_file(split_json)
        expected_seed_meta = {
            **cfg_meta,
            "split_json_sha1": split_json_sha1,
            "seed": int(seed),
        }

        d, found_metrics = _try_load_metrics(metrics_json, eval_json, expected_meta=expected_seed_meta)
        if d is not None:
            print(f"[info] skip seed={seed}: found metrics -> {found_metrics}")
        else:
            can_reuse_ckpt = False
            if ckpt_best.exists() and run_sig_json.exists():
                try:
                    sig = _load_json(run_sig_json)
                    can_reuse_ckpt = _signature_matches(sig, expected_seed_meta)
                except Exception:
                    can_reuse_ckpt = False

            if not can_reuse_ckpt:
                cmd_train = [
                    str(args.python),
                    "train.py",
                    "--dataset_cfg", str(dataset_cfg),
                    "--model_cfg", str(model_cfg),
                    "--train_cfg", str(train_cfg),
                    "--split_json", str(split_json),
                    "--data_root", str(args.data_root),
                    "--seed", str(seed),
                    "--out_dir", str(out_dir),
                    "--num_workers", str(int(num_workers)),
                ]
                if args.amp:
                    cmd_train.append("--amp")
                _run(cmd_train)
            else:
                print("[info] skip train: found compatible checkpoint/signature", ckpt_best)

            d, found_metrics = _try_load_metrics(metrics_json, eval_json, expected_meta=expected_seed_meta)
            if d is None:
                cmd_eval = [
                    str(args.python),
                    "eval.py",
                    "--dataset_cfg", str(dataset_cfg),
                    "--model_cfg", str(model_cfg),
                    "--train_cfg", str(train_cfg),
                    "--split_json", str(split_json),
                    "--data_root", str(args.data_root),
                    "--seed", str(seed),
                    "--ckpt", str(ckpt_best),
                    "--ckpt_key", "model",
                    "--batch_size", str(int(eval_batch_size)),
                    "--num_workers", str(int(num_workers)),
                    "--log_interval", str(int(eval_log_interval)),
                    "--out", str(eval_json),
                ]
                if args.amp:
                    cmd_eval.append("--amp")
                _run_capture(cmd_eval)
                d, found_metrics = _try_load_metrics(metrics_json, eval_json, expected_meta=expected_seed_meta)
            if d is None:
                raise SystemExit(f"[ERROR] missing valid metrics after train/eval for seed={seed}: {out_dir}")

        v = _pick_split_dict(d, "val")
        t = _pick_split_dict(d, "test")
        meta_seed = d.get("meta", {}) if isinstance(d.get("meta", {}), dict) else {}

        per_seed["val"][str(seed)] = {k: float(v[k]) for k in keys}
        per_seed["test"][str(seed)] = {k: float(t[k]) for k in keys}
        per_seed_full["val"][str(seed)] = dict(v)
        per_seed_full["test"][str(seed)] = dict(t)
        per_seed_full["meta"][str(seed)] = dict(meta_seed)
        for k in keys:
            val_vals[k].append(float(v[k]))
            test_vals[k].append(float(t[k]))
        t_sec = d.get("time_sec", meta_seed.get("time_sec"))
        if isinstance(t_sec, (int, float)):
            run_time_vals.append(float(t_sec))
        p_total = meta_seed.get("num_params_total")
        if isinstance(p_total, (int, float)):
            params_total_vals.append(float(p_total))
        p_train = meta_seed.get("num_params_trainable")
        if isinstance(p_train, (int, float)):
            params_trainable_vals.append(float(p_train))

    def mean_std(xs: List[float]):
        a = np.asarray(xs, dtype=float)
        return float(a.mean()), float(a.std(ddof=0))

    mean = {"val": {}, "test": {}}
    std = {"val": {}, "test": {}}
    for k in keys:
        mean["val"][k], std["val"][k] = mean_std(val_vals[k])
        mean["test"][k], std["test"][k] = mean_std(test_vals[k])

    per_class_summary: Dict[str, Dict[str, List[float]]] = {}
    for split_name in ("val", "test"):
        arrs: List[np.ndarray] = []
        valid = True
        for seed in seeds:
            blk = per_seed_full[split_name].get(str(seed), {})
            if not isinstance(blk, dict):
                valid = False
                break
            pc = blk.get("per_class_acc")
            if not isinstance(pc, list) or len(pc) == 0:
                valid = False
                break
            arrs.append(np.asarray(pc, dtype=np.float64).reshape(-1))
        if valid and arrs and all(a.shape == arrs[0].shape for a in arrs):
            mat = np.stack(arrs, axis=0)
            per_class_summary[split_name] = {
                "mean": np.mean(mat, axis=0).astype(float).tolist(),
                "std": np.std(mat, axis=0, ddof=0).astype(float).tolist(),
            }

    mean_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "per_seed": per_seed,
        "per_seed_full": per_seed_full,
        "mean": mean,
        "std": std,
        "meta": {
            "dataset": dataset,
            "split_tag": args.split_tag,
            "seed_spec": str(args.seeds),
            "num_seeds": int(n_seeds),
            "seeds": [int(x) for x in seeds],
            "protocol": f"RANDOM only; {n_seeds} seeds ({args.seeds}); STRICT split_json indices; single-pass eval; mean±std reported",
            "dataset_cfg": str(dataset_cfg),
            "model_cfg": str(model_cfg),
            "train_cfg": str(train_cfg),
            "dataset_cfg_sha1": _sha1_file(dataset_cfg),
            "model_cfg_sha1": _sha1_file(model_cfg),
            "train_cfg_sha1": _sha1_file(train_cfg),
        },
    }
    if run_time_vals:
        out["time_sec"] = {
            "per_seed": run_time_vals,
            "mean": float(np.mean(np.asarray(run_time_vals, dtype=np.float64))),
            "std": float(np.std(np.asarray(run_time_vals, dtype=np.float64), ddof=0)),
        }
    if params_total_vals:
        out["params_total"] = {
            "per_seed": params_total_vals,
            "mean": float(np.mean(np.asarray(params_total_vals, dtype=np.float64))),
            "std": float(np.std(np.asarray(params_total_vals, dtype=np.float64), ddof=0)),
        }
    if params_trainable_vals:
        out["params_trainable"] = {
            "per_seed": params_trainable_vals,
            "mean": float(np.mean(np.asarray(params_trainable_vals, dtype=np.float64))),
            "std": float(np.std(np.asarray(params_trainable_vals, dtype=np.float64), ddof=0)),
        }
    if per_class_summary:
        out["per_class_acc"] = per_class_summary
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
    md.append(f"# {dataset} ({args.split_tag}) {n_seeds}-seed summary\n")
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
