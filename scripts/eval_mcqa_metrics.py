#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_mcqa_metrics.py
- Read JSONL (one dict per line)
- Auto/manual extraction of gold and pred (supports A/B/C/D or 1~4 or 0~3)
- Compute:
  - ACC
  - Per-class Precision / Recall / F1 / Support
  - Macro Avg
  - Weighted Avg (P/R/F1 all reported)
  - Micro Avg (for single-label multi-class equals ACC; even invalid predictions are counted as UNK)
  - Confusion Matrix (rows=Gold, cols=Pred)
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------
# Parse labels: supports A/B/C/D, 1~4, 0~3
# -----------------------------
DEFAULT_LABELS = ["A", "B", "C", "D"]

_pat_letter = re.compile(r"\b([A-D])\b", re.IGNORECASE)
_pat_digit14 = re.compile(r"\b([1-4])\b")
_pat_digit03 = re.compile(r"\b([0-3])\b")

def parse_label(x: Any, labels: List[str]) -> Optional[str]:
    """
    Parse arbitrary input into one of the labels (e.g., A/B/C/D). Return None if parsing fails.
    """
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        # 1~4 or 0~3
        if 1 <= int(x) <= len(labels):
            return labels[int(x) - 1]
        if 0 <= int(x) < len(labels):
            return labels[int(x)]
        return None

    s = str(x).strip()
    if not s:
        return None

    su = s.upper()

    # Direct letter
    if su in labels:
        return su

    # Direct digit
    if su.isdigit():
        v = int(su)
        if 1 <= v <= len(labels):
            return labels[v - 1]
        if 0 <= v < len(labels):
            return labels[v]
        return None

    # Search within text
    m = _pat_letter.search(su)
    if m and m.group(1) in labels:
        return m.group(1)

    m = _pat_digit14.search(su)
    if m:
        v = int(m.group(1))
        return labels[v - 1]

    m = _pat_digit03.search(su)
    if m:
        v = int(m.group(1))
        return labels[v]

    # Fallback: scan the first few characters
    for ch in su[:12]:
        if ch in labels:
            return ch

    return None


# -----------------------------
# JSON dict value retrieval: supports dotted path (a.b.c)
# -----------------------------
def get_by_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


# -----------------------------
# Auto-detect gold/pred field names
# -----------------------------
GOLD_CANDIDATES = [
    "gold", "gt", "label", "answer", "cop", "ground_truth", "y_true", "target", "correct"
]
PRED_CANDIDATES = [
    "pred", "prediction", "predicted", "output", "choice", "y_pred",
    "base_prediction", "rag_prediction", "model_pred"
]

def autodetect_key(keys: List[str], candidates: List[str]) -> Optional[str]:
    ks = set(keys)
    for c in candidates:
        if c in ks:
            return c
    return None


# -----------------------------
# Metric computation
# -----------------------------
def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0

def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def compute_report(golds: List[str], preds: List[str], labels: List[str]) -> Dict[str, Any]:
    """
    labels: target label set (typically A/B/C/D)
    golds/preds: same length. preds may include values not in labels (treated as UNK)
    """
    assert len(golds) == len(preds)
    n = len(golds)

    # Distribution stats (count labels + UNK)
    pred_cnt = Counter(preds)
    gold_cnt = Counter(golds)

    # --- per-class stats (computed only over labels) ---
    # confusion matrix: rows=gold, cols=pred (includes labels + UNK)
    cols = labels[:]  # start with A/B/C/D
    if any(p not in labels for p in preds):
        cols = cols + ["UNK"]

    # Initialize CM
    cm = {g: {c: 0 for c in cols} for g in labels}
    for g, p in zip(golds, preds):
        col = p if p in labels else "UNK"
        if g in labels:
            cm[g][col] += 1

    # TP/FP/FN
    per = []  # (label, prec, recall, f1, support, tp)
    for lab in labels:
        support = gold_cnt.get(lab, 0)
        tp = cm[lab].get(lab, 0)
        # fp: predicted as lab but gold is not lab (includes other gold classes)
        fp = sum(cm[g].get(lab, 0) for g in labels if g != lab)
        # fn: gold is lab but pred is not lab (includes pred=UNK)
        fn = support - tp
        p, r, f1 = prf1(tp, fp, fn)
        per.append((lab, p, r, f1, support, tp))

    # --- macro/weighted ---
    ps = np.array([x[1] for x in per], dtype=float)
    rs = np.array([x[2] for x in per], dtype=float)
    fs = np.array([x[3] for x in per], dtype=float)
    ws = np.array([x[4] for x in per], dtype=float)
    W = float(ws.sum()) if float(ws.sum()) > 0 else 1.0

    macro_p = float(ps.mean()) if len(ps) else 0.0
    macro_r = float(rs.mean()) if len(rs) else 0.0
    macro_f = float(fs.mean()) if len(fs) else 0.0

    weighted_p = float((ps * ws).sum() / W) if len(ps) else 0.0
    weighted_r = float((rs * ws).sum() / W) if len(rs) else 0.0
    weighted_f = float((fs * ws).sum() / W) if len(fs) else 0.0

    # --- micro & acc ---
    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    acc = safe_div(correct, n)

    # micro: single-label multi-class -> micro P=R=F1=ACC (one prediction per sample)
    micro_p = acc
    micro_r = acc
    micro_f = acc

    return {
        "n": n,
        "labels": labels,
        "pred_cnt": pred_cnt,
        "gold_cnt": gold_cnt,
        "per": per,
        "macro": (macro_p, macro_r, macro_f),
        "weighted": (weighted_p, weighted_r, weighted_f),
        "micro": (micro_p, micro_r, micro_f),
        "acc": acc,
        "cm_cols": cols,
        "cm": cm,
        "correct": correct,
    }


def print_report(rep: Dict[str, Any]) -> None:
    labels = rep["labels"]
    pred_cnt: Counter = rep["pred_cnt"]
    gold_cnt: Counter = rep["gold_cnt"]
    per = rep["per"]
    macro_p, macro_r, macro_f = rep["macro"]
    weighted_p, weighted_r, weighted_f = rep["weighted"]
    micro_p, micro_r, micro_f = rep["micro"]
    acc = rep["acc"]

    # Distributions
    print("\n=== Distribution ===")
    # Print in labels + UNK order
    order = labels + (["UNK"] if "UNK" in pred_cnt else [])
    print("Pred count:", {k: int(pred_cnt.get(k, 0)) for k in order})
    print("Gold count:", {k: int(gold_cnt.get(k, 0)) for k in labels})

    # Per-class
    print("\n=== Per-class ===")
    print("Class  |  Prec   Recall   F1     Support   Correct/Gold")
    for lab, p, r, f1, sup, tp in per:
        print(f"  {lab:<4} |  {p:0.4f}  {r:0.4f}  {f1:0.4f}   {sup:>7}     {tp}/{sup}")

    # Averages
    print("\n=== Averages ===")
    print(f"Macro-Avg  P={macro_p:0.4f}  R={macro_r:0.4f}  F1={macro_f:0.4f}")
    print(f"Weighted   P={weighted_p:0.4f}  R={weighted_r:0.4f}  F1={weighted_f:0.4f}")
    print(f"Micro-Avg  P={micro_p:0.4f}  R={micro_r:0.4f}  F1={micro_f:0.4f}")
    print(f"ACC        = {acc:0.4f}")

    # Confusion matrix
    cm = rep["cm"]
    cols = rep["cm_cols"]

    print("\n=== Confusion Matrix (rows=Gold, cols=Pred) ===")
    header = "      " + "  ".join([f"{c:>3}" for c in cols])
    print(header)
    for g in labels:
        row = [cm[g].get(c, 0) for c in cols]
        print(f"{g} | " + "  ".join([f"{v:>5}" for v in row]))


# -----------------------------
# File reading & main flow
# -----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input jsonl file path")
    ap.add_argument("--gold_key", default="", help="Gold field name (supports dotted path, e.g., a.b.c)")
    ap.add_argument("--pred_key", default="", help="Pred field name (supports dotted path)")
    ap.add_argument("--labels", default="ABCD", help="Class label order, default ABCD")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N items (0=all)")
    args = ap.parse_args()

    path = args.inp
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    labels = list(args.labels.strip().upper())
    if not labels:
        labels = DEFAULT_LABELS

    data = read_jsonl(path)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    if not data:
        print("Empty file.")
        return

    # Auto-detect fields
    keys0 = list(data[0].keys())
    gold_key = args.gold_key.strip() or autodetect_key(keys0, GOLD_CANDIDATES)
    pred_key = args.pred_key.strip() or autodetect_key(keys0, PRED_CANDIDATES)

    if not gold_key or not pred_key:
        print("Unable to auto-detect gold/pred field names.")
        print("Current sample keys:", keys0)
        print("Please specify manually: --gold_key xxx --pred_key yyy")
        return

    golds: List[str] = []
    preds: List[str] = []
    dropped = 0

    for obj in data:
        g_raw = get_by_path(obj, gold_key) if "." in gold_key else obj.get(gold_key)
        p_raw = get_by_path(obj, pred_key) if "." in pred_key else obj.get(pred_key)

        g = parse_label(g_raw, labels)
        p = parse_label(p_raw, labels)

        # If gold parsing fails: cannot score this sample (drop it)
        if g is None:
            dropped += 1
            continue

        # If pred parsing fails: mark as UNK (still counted in overall ACC / micro)
        if p is None:
            p = "UNK"

        golds.append(g)
        preds.append(p)

    if not golds:
        print("No evaluable samples (all gold labels failed to parse).")
        return

    if dropped:
        print(f"[Warn] dropped {dropped} samples because gold label could not be parsed.")

    rep = compute_report(golds, preds, labels)
    print_report(rep)

    # Extra field info
    print(f"\n[Info] using gold_key='{gold_key}', pred_key='{pred_key}', evaluated={rep['n']}")

if __name__ == "__main__":
    main()