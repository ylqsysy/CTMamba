#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np


def oa_aa_kappa(conf: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    conf = np.asarray(conf, dtype=np.int64)
    total = conf.sum()
    correct = np.trace(conf)
    oa = float(correct / (total + 1e-12))

    per_class = conf.diagonal() / np.maximum(conf.sum(axis=1), 1)
    aa = float(np.mean(per_class))

    row_marg = conf.sum(axis=1)
    col_marg = conf.sum(axis=0)
    pe = float((row_marg @ col_marg) / ((total + 1e-12) ** 2))
    kappa = float((oa - pe) / (1.0 - pe + 1e-12))
    return oa, aa, kappa, per_class.astype(np.float64), conf


def classification_report(conf: np.ndarray) -> Dict[str, Any]:
    """
    Backward-compatible helper for older training code.

    Returns OA/AA/Kappa + per-class accuracy + confusion matrix.
    """
    oa, aa, kappa, per_class, cm = oa_aa_kappa(conf)
    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "per_class_acc": per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }
