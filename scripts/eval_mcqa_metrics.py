#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_mcqa_metrics.py
- 读取 JSONL（每行一个 dict）
- 自动/手动提取 gold 与 pred（支持 A/B/C/D 或 1~4 或 0~3）
- 计算：
  - ACC
  - Per-class Precision / Recall / F1 / Support
  - Macro Avg
  - Weighted Avg（P/R/F1 都给）
  - Micro Avg（单标签多分类下等于 ACC；即使出现无效预测也会按 UNK 计入）
  - Confusion Matrix（rows=Gold, cols=Pred）
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------
# 解析标签：支持 A/B/C/D, 1~4, 0~3
# -----------------------------
DEFAULT_LABELS = ["A", "B", "C", "D"]

_pat_letter = re.compile(r"\b([A-D])\b", re.IGNORECASE)
_pat_digit14 = re.compile(r"\b([1-4])\b")
_pat_digit03 = re.compile(r"\b([0-3])\b")

def parse_label(x: Any, labels: List[str]) -> Optional[str]:
    """
    把任意输入解析成 labels 中的一个（如 A/B/C/D），解析失败返回 None。
    """
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        # 1~4 或 0~3
        if 1 <= int(x) <= len(labels):
            return labels[int(x) - 1]
        if 0 <= int(x) < len(labels):
            return labels[int(x)]
        return None

    s = str(x).strip()
    if not s:
        return None

    su = s.upper()

    # 直接是字母
    if su in labels:
        return su

    # 直接是数字
    if su.isdigit():
        v = int(su)
        if 1 <= v <= len(labels):
            return labels[v - 1]
        if 0 <= v < len(labels):
            return labels[v]
        return None

    # 文本里搜
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

    # 兜底：前若干字符里找
    for ch in su[:12]:
        if ch in labels:
            return ch

    return None


# -----------------------------
# JSON dict 取值：支持 dotted path (a.b.c)
# -----------------------------
def get_by_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


# -----------------------------
# 自动猜 gold/pred 字段名
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
# 计算指标
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
    labels：目标标签集合（一般是 A/B/C/D）
    golds/preds：长度相同。pred 允许出现不在 labels 的值（会当作 UNK）
    """
    assert len(golds) == len(preds)
    n = len(golds)

    # 统计分布（只统计 labels + UNK）
    pred_cnt = Counter(preds)
    gold_cnt = Counter(golds)

    # --- per-class 统计（只对 labels 计算）---
    # confusion matrix：rows=gold, cols=pred（包含 labels + UNK）
    cols = labels[:]  # 先放 A/B/C/D
    if any(p not in labels for p in preds):
        cols = cols + ["UNK"]

    # 初始化 CM
    cm = {g: {c: 0 for c in cols} for g in labels}
    for g, p in zip(golds, preds):
        col = p if p in labels else "UNK"
        if g in labels:
            cm[g][col] += 1

    # TP/FP/FN 计算
    per = []  # (label, prec, recall, f1, support, tp)
    # 先准备：每个 label 的 support / tp / fp / fn
    for lab in labels:
        support = gold_cnt.get(lab, 0)
        tp = cm[lab].get(lab, 0)
        # fp：预测为 lab 但 gold 不是 lab（含其他 gold 类）
        fp = sum(cm[g].get(lab, 0) for g in labels if g != lab)
        # fn：gold 为 lab 但 pred 不是 lab（含 pred=UNK）
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

    # micro：单标签多分类 -> micro P=R=F1=ACC（按“每样本1个预测”定义）
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

    # 分布
    print("\n=== Distribution ===")
    # 只按 labels + UNK 排序打印
    order = labels + (["UNK"] if "UNK" in pred_cnt else [])
    print("Pred count:", {k: int(pred_cnt.get(k, 0)) for k in order})
    print("Gold count:", {k: int(gold_cnt.get(k, 0)) for k in labels})

    # per-class
    print("\n=== Per-class ===")
    print("Class  |  Prec   Recall   F1     Support   Correct/Gold")
    for lab, p, r, f1, sup, tp in per:
        print(f"  {lab:<4} |  {p:0.4f}  {r:0.4f}  {f1:0.4f}   {sup:>7}     {tp}/{sup}")

    # averages
    print("\n=== Averages ===")
    print(f"Macro-Avg  P={macro_p:0.4f}  R={macro_r:0.4f}  F1={macro_f:0.4f}")
    print(f"Weighted   P={weighted_p:0.4f}  R={weighted_r:0.4f}  F1={weighted_f:0.4f}")
    print(f"Micro-Avg  P={micro_p:0.4f}  R={micro_r:0.4f}  F1={micro_f:0.4f}")
    print(f"ACC        = {acc:0.4f}")

    # confusion matrix
    cm = rep["cm"]
    cols = rep["cm_cols"]

    print("\n=== Confusion Matrix (rows=Gold, cols=Pred) ===")
    header = "      " + "  ".join([f"{c:>3}" for c in cols])
    print(header)
    for g in labels:
        row = [cm[g].get(c, 0) for c in cols]
        print(f"{g} | " + "  ".join([f"{v:>5}" for v in row]))


# -----------------------------
# 读文件 & 主流程
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
    ap.add_argument("--in", dest="inp", required=True, help="输入 jsonl 文件路径")
    ap.add_argument("--gold_key", default="", help="gold 字段名（支持 dotted path，如 a.b.c）")
    ap.add_argument("--pred_key", default="", help="pred 字段名（支持 dotted path）")
    ap.add_argument("--labels", default="ABCD", help="类别标签顺序，默认 ABCD")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 条（0=全量）")
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

    # 自动识别字段
    keys0 = list(data[0].keys())
    gold_key = args.gold_key.strip() or autodetect_key(keys0, GOLD_CANDIDATES)
    pred_key = args.pred_key.strip() or autodetect_key(keys0, PRED_CANDIDATES)

    if not gold_key or not pred_key:
        print("无法自动识别 gold/pred 字段名。")
        print("当前样本 keys:", keys0)
        print("请手动指定：--gold_key xxx --pred_key yyy")
        return

    golds: List[str] = []
    preds: List[str] = []
    dropped = 0

    for obj in data:
        g_raw = get_by_path(obj, gold_key) if "." in gold_key else obj.get(gold_key)
        p_raw = get_by_path(obj, pred_key) if "." in pred_key else obj.get(pred_key)

        g = parse_label(g_raw, labels)
        p = parse_label(p_raw, labels)

        # gold 解析失败：这条没法算（直接丢弃）
        if g is None:
            dropped += 1
            continue

        # pred 解析失败：记为 UNK（仍计入总体 ACC / micro）
        if p is None:
            p = "UNK"

        golds.append(g)
        preds.append(p)

    if not golds:
        print("没有可评测样本（gold 全解析失败）。")
        return

    if dropped:
        print(f"[Warn] dropped {dropped} samples because gold label could not be parsed.")

    rep = compute_report(golds, preds, labels)
    print_report(rep)

    # 额外提示字段
    print(f"\n[Info] using gold_key='{gold_key}', pred_key='{pred_key}', evaluated={rep['n']}")

if __name__ == "__main__":
    main()
