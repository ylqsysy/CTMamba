#!/usr/bin/env python3
"""Run ablation suites and aggregate mean/std tables."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DATASETS = ["pavia_university", "houston2013", "hanchuan", "honghu"]


def _run(cmd: List[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _load_yaml(p: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _save_yaml(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    txt = yaml.safe_dump(obj, sort_keys=False, allow_unicode=False)
    p.write_text(txt, encoding="utf-8")


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _infer_paths(
    dataset: str,
    *,
    dataset_cfg_dir: Path,
    model_cfg_dir: Path,
    train_cfg_dir: Path,
) -> Tuple[Path, Path, Path]:
    d = dataset_cfg_dir / f"{dataset}.yaml"

    model_map = {
        "pavia_university": "vssm3d_pu.yaml",
        "houston2013": "vssm3d_houston2013.yaml",
        "hanchuan": "vssm3d_hanchuan.yaml",
        "honghu": "vssm3d_honghu.yaml",
    }
    m = model_cfg_dir / model_map[dataset]

    train_map = {
        "pavia_university": "pu.yaml",
        "houston2013": "houston2013.yaml",
        "hanchuan": "hanchuan.yaml",
        "honghu": "honghu.yaml",
    }
    t = train_cfg_dir / train_map[dataset]
    return d, m, t


def _sanitize_tag(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.+-]+", "_", str(s).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "variant"


def _sanitize_dataset_tag(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "dataset"


def _parse_int_list(spec: str) -> List[int]:
    vals: List[int] = []
    for x in str(spec).split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise SystemExit("[ERROR] empty integer list.")
    return vals


def _parse_float_list(spec: str) -> List[float]:
    vals: List[float] = []
    for x in str(spec).split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise SystemExit("[ERROR] empty float list.")
    return vals


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


def _parse_datasets(spec: str) -> List[str]:
    s = str(spec).strip().lower()
    if s == "all":
        return list(DATASETS)
    vals = [x.strip() for x in str(spec).split(",") if x.strip()]
    if not vals:
        raise SystemExit("[ERROR] empty --dataset.")
    bad = [x for x in vals if x not in DATASETS]
    if bad:
        raise SystemExit(f"[ERROR] unsupported dataset(s): {bad}; supported={DATASETS}")
    return vals


def _dedup_variants(
    variants: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    seen = set()
    out: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for name, m_ovr, t_ovr in variants:
        key = (json.dumps(m_ovr, sort_keys=True), json.dumps(t_ovr, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, m_ovr, t_ovr))
    return out


def _core_variants(base_model_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    base_pool = str(base_model_cfg.get("pool", "mean")).strip().lower()
    variants: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = [
        ("full", {}, {}),
        ("no_coords", {"append_coords": False}, {}),
        ("no_spec_branch", {"use_spec_branch": False}, {}),
        ("no_spec_cross_attn", {"use_spec_cross_attn": False}, {}),
        ("no_tokenizer", {"token_bands": 0}, {}),
        ("pool_mean", {"pool": "mean"}, {}),
    ]
    if base_pool != "center_attn":
        variants.append(("pool_center_attn", {"pool": "center_attn"}, {}))
    return _dedup_variants(variants)


def _fmt(v: Any, digits: int = 4) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return ""


def _write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_tables(records: List[Dict[str, Any]], out_base: Path, n_seeds: int) -> None:
    if not records:
        return

    exp_names = sorted({str(r["exp_name"]) for r in records})
    for exp_name in exp_names:
        rows = [r for r in records if str(r["exp_name"]) == exp_name]
        rows = sorted(rows, key=lambda x: (str(x["dataset"]), str(x["variant"])))

        csv_lines = [
            "dataset,exp_name,variant,seeds,val_OA,val_OA_std,val_AA,val_AA_std,val_Kappa,val_Kappa_std,test_OA,test_OA_std,test_AA,test_AA_std,test_Kappa,test_Kappa_std,mean_metrics_json"
        ]
        md_lines = [
            f"# {exp_name} summary (mean over {n_seeds} seeds)",
            "",
            "| dataset | variant | test_OA | test_OA_std | test_AA | test_AA_std | test_Kappa | test_Kappa_std |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]

        for r in rows:
            p = Path(str(r["mean_metrics_path"]))
            if p.exists():
                d = _load_json(p)
                mean = d.get("mean", {})
                std = d.get("std", {})
                val_m = mean.get("val", {})
                val_s = std.get("val", {})
                test_m = mean.get("test", {})
                test_s = std.get("test", {})
                csv_lines.append(
                    ",".join(
                        [
                            str(r["dataset"]),
                            str(r["exp_name"]),
                            str(r["variant"]),
                            str(n_seeds),
                            _fmt(val_m.get("OA"), 6),
                            _fmt(val_s.get("OA"), 6),
                            _fmt(val_m.get("AA"), 6),
                            _fmt(val_s.get("AA"), 6),
                            _fmt(val_m.get("Kappa"), 6),
                            _fmt(val_s.get("Kappa"), 6),
                            _fmt(test_m.get("OA"), 6),
                            _fmt(test_s.get("OA"), 6),
                            _fmt(test_m.get("AA"), 6),
                            _fmt(test_s.get("AA"), 6),
                            _fmt(test_m.get("Kappa"), 6),
                            _fmt(test_s.get("Kappa"), 6),
                            str(p),
                        ]
                    )
                )
                md_lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(r["dataset"]),
                            str(r["variant"]),
                            _fmt(test_m.get("OA")),
                            _fmt(test_s.get("OA")),
                            _fmt(test_m.get("AA")),
                            _fmt(test_s.get("AA")),
                            _fmt(test_m.get("Kappa")),
                            _fmt(test_s.get("Kappa")),
                        ]
                    )
                    + " |"
                )
            else:
                csv_lines.append(
                    ",".join(
                        [
                            str(r["dataset"]),
                            str(r["exp_name"]),
                            str(r["variant"]),
                            str(n_seeds),
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            str(p),
                        ]
                    )
                )
                md_lines.append(
                    "| "
                    + " | ".join(
                        [str(r["dataset"]), str(r["variant"]), "MISSING", "", "", "", "", ""]
                    )
                    + " |"
                )

        _write_lines(out_base / f"summary_{exp_name}_mean{n_seeds}.csv", csv_lines)
        _write_lines(out_base / f"summary_{exp_name}_mean{n_seeds}.md", md_lines)

    datasets = sorted({str(r["dataset"]) for r in records})
    for dataset in datasets:
        ds_rows = [r for r in records if str(r["dataset"]) == dataset]
        exp_names_ds = sorted({str(r["exp_name"]) for r in ds_rows})
        for exp_name in exp_names_ds:
            rows = [r for r in ds_rows if str(r["exp_name"]) == exp_name]
            rows = sorted(rows, key=lambda x: str(x["variant"]))

            csv_lines = [
                "variant,seeds,test_OA,test_OA_std,test_AA,test_AA_std,test_Kappa,test_Kappa_std,mean_metrics_json"
            ]
            for r in rows:
                p = Path(str(r["mean_metrics_path"]))
                if p.exists():
                    d = _load_json(p)
                    mean = d.get("mean", {})
                    std = d.get("std", {})
                    test_m = mean.get("test", {})
                    test_s = std.get("test", {})
                    csv_lines.append(
                        ",".join(
                            [
                                str(r["variant"]),
                                str(n_seeds),
                                _fmt(test_m.get("OA"), 6),
                                _fmt(test_s.get("OA"), 6),
                                _fmt(test_m.get("AA"), 6),
                                _fmt(test_s.get("AA"), 6),
                                _fmt(test_m.get("Kappa"), 6),
                                _fmt(test_s.get("Kappa"), 6),
                                str(p),
                            ]
                        )
                    )
                else:
                    csv_lines.append(",".join([str(r["variant"]), str(n_seeds), "", "", "", "", "", "", str(p)]))

            _write_lines(
                out_base / dataset / exp_name / f"summary_mean{n_seeds}.csv",
                csv_lines,
            )


def _resolve_path(path_like: str) -> Path:
    p = Path(str(path_like))
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _mean_metrics_complete(mean_json: Path, seeds: List[int]) -> bool:
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
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="all",
        help="Dataset name or comma-separated names or 'all'.",
    )
    ap.add_argument("--exp", default="suite", choices=["core", "patch", "lr", "all", "suite"])
    ap.add_argument("--split_tag", default="random")
    ap.add_argument("--seeds", default="0", help="Ablation seed spec. Default: seed0 only.")
    ap.add_argument("--min_seed_count", type=int, default=1, help="Optional minimum seed count constraint.")
    ap.add_argument("--allow_fewer_seeds", action="store_true", help="Allow seed count below --min_seed_count.")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_base", default="outputs/ablations")
    ap.add_argument("--dataset_cfg_dir", default="configs/datasets")
    ap.add_argument("--model_cfg_dir", default="configs/model")
    ap.add_argument("--train_cfg_dir", default="configs/train")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help=">=0 to force value; -1 to follow each dataset train_cfg(num_workers).",
    )
    ap.add_argument(
        "--eval_batch_size",
        type=int,
        default=-1,
        help=">0 to force value; <=0 to follow each dataset train_cfg(eval_batch_size).",
    )
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--patch_sizes", default="15")
    ap.add_argument("--lrs", default="8e-5,1.3e-4,1.8e-4,2.3e-4,2.8e-4")
    ap.add_argument("--no_summary", action="store_true")
    ap.add_argument("--force", action="store_true", help="Force rerun even if variant mean_metrics already complete.")
    args = ap.parse_args()

    datasets = _parse_datasets(str(args.dataset))
    seeds = _parse_seeds(str(args.seeds))
    n_seeds = len(seeds)

    if (not bool(args.allow_fewer_seeds)) and n_seeds < int(args.min_seed_count):
        raise SystemExit(
            f"[ERROR] got {n_seeds} seeds (< min_seed_count={int(args.min_seed_count)}). "
            "Increase --seeds or pass --allow_fewer_seeds."
        )

    do_core = args.exp in ("core", "all", "suite")
    do_patch = args.exp in ("patch", "all")
    do_lr = args.exp in ("lr", "all", "suite")

    out_base = Path(args.out_base)
    records: List[Dict[str, Any]] = []

    print("[info] datasets    =", datasets)
    print("[info] exp         =", args.exp)
    print("[info] seeds       =", seeds)
    print("[info] n_seeds     =", n_seeds)
    print("[info] out_base    =", out_base)

    dataset_cfg_dir = _resolve_path(args.dataset_cfg_dir)
    model_cfg_dir = _resolve_path(args.model_cfg_dir)
    train_cfg_dir = _resolve_path(args.train_cfg_dir)

    for dataset in datasets:
        dataset_cfg_path, base_model_cfg_path, base_train_cfg_path = _infer_paths(
            dataset,
            dataset_cfg_dir=dataset_cfg_dir,
            model_cfg_dir=model_cfg_dir,
            train_cfg_dir=train_cfg_dir,
        )
        if not dataset_cfg_path.exists():
            raise SystemExit(f"[ERROR] missing dataset_cfg: {dataset_cfg_path}")
        if not base_model_cfg_path.exists():
            raise SystemExit(f"[ERROR] missing model_cfg: {base_model_cfg_path}")
        if not base_train_cfg_path.exists():
            raise SystemExit(f"[ERROR] missing train_cfg: {base_train_cfg_path}")

        base_model_cfg = _load_yaml(base_model_cfg_path)
        base_train_cfg = _load_yaml(base_train_cfg_path)
        dataset_tag = _sanitize_dataset_tag(dataset)

        def run_variant(
            exp_name: str,
            variant_name: str,
            model_override: Dict[str, Any],
            train_override: Dict[str, Any],
        ) -> None:
            mcfg = dict(base_model_cfg)
            tcfg = dict(base_train_cfg)
            mcfg.update(model_override)
            tcfg.update(train_override)

            variant_tag = _sanitize_tag(variant_name)
            variant_out = out_base / dataset / exp_name / variant_tag
            mean_metrics_path = (
                variant_out / f"{dataset_tag}_mean{n_seeds}" / "mean_metrics.json"
            )

            if (not bool(args.force)) and _mean_metrics_complete(mean_metrics_path, seeds):
                print(f"[info] skip variant: {dataset}/{exp_name}/{variant_tag} (complete)")
                records.append(
                    {
                        "dataset": dataset,
                        "exp_name": exp_name,
                        "variant": variant_tag,
                        "mean_metrics_path": str(mean_metrics_path),
                    }
                )
                return

            with tempfile.TemporaryDirectory(prefix=f"abl_{dataset}_{exp_name}_{variant_tag}_") as tmpdir:
                tmp = Path(tmpdir)
                model_cfg_path = tmp / "model.yaml"
                train_cfg_path = tmp / "train.yaml"
                _save_yaml(model_cfg_path, mcfg)
                _save_yaml(train_cfg_path, tcfg)

                cmd = [
                    str(args.python),
                    "run_multiseed.py",
                    "--dataset",
                    dataset,
                    "--split_tag",
                    str(args.split_tag),
                    "--seeds",
                    str(args.seeds),
                    "--data_root",
                    str(args.data_root),
                    "--out_base",
                    str(variant_out),
                    "--num_workers",
                    str(int(args.num_workers)),
                    "--eval_batch_size",
                    str(int(args.eval_batch_size)),
                    "--python",
                    str(args.python),
                    "--dataset_cfg_path",
                    str(dataset_cfg_path),
                    "--model_cfg_path",
                    str(model_cfg_path),
                    "--train_cfg_path",
                    str(train_cfg_path),
                ]
                if args.amp:
                    cmd.append("--amp")
                if args.force:
                    cmd.append("--force")
                _run(cmd)

            records.append(
                {
                    "dataset": dataset,
                    "exp_name": exp_name,
                    "variant": variant_tag,
                    "mean_metrics_path": str(mean_metrics_path),
                }
            )

        if do_core:
            for name, m_ovr, t_ovr in _core_variants(base_model_cfg):
                run_variant("core_modules", name, m_ovr, t_ovr)

        if do_patch:
            patch_sizes = _parse_int_list(args.patch_sizes)
            for ps in patch_sizes:
                if ps <= 0 or ps % 2 == 0:
                    raise SystemExit(f"[ERROR] patch_size must be positive odd, got {ps}")
                run_variant("patch_size", f"patch{ps}", {"patch_size": int(ps)}, {})

        if do_lr:
            lrs = _parse_float_list(args.lrs)
            run_variant("lr", "lr_base", {}, {})
            for lr in lrs:
                if lr <= 0:
                    raise SystemExit(f"[ERROR] lr must be > 0, got {lr}")
                run_variant("lr", f"lr_{_sanitize_tag(f'{lr:.8g}')}", {}, {"lr": float(lr)})

    if not bool(args.no_summary):
        _write_summary_tables(records, out_base=out_base, n_seeds=n_seeds)
        print("[DONE] wrote summary tables under", out_base)


if __name__ == "__main__":
    main()
