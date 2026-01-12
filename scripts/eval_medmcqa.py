# scripts/eval_medmcqa.py  (v2)
import argparse, json, collections, os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

LETTER = "ABCD"

def cop_to_letter(cop):
    if cop is None: return None
    s = str(cop).strip().lower()
    if s in ("a","b","c","d"): return s.upper()
    if s.isdigit():
        k = int(s)
        if 1 <= k <= 4: return LETTER[k-1]
        if 0 <= k <= 3: return LETTER[k]
    return None

def load_labels(raw_jsonl):
    labels = {}
    with open(raw_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            o = json.loads(line)
            y = cop_to_letter(o.get("cop", o.get("answer", o.get("label"))))
            if y is not None:
                labels[i] = y
    return labels

def load_preds(pred_jsonl):
    preds = {}
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            pid = o.get("id")
            p = (o.get("pred_letter") or o.get("pred") or o.get("letter") or "").strip().upper()[:1]
            preds[pid] = p if p in LETTER else None  # None indicates an invalid prediction
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    labels = load_labels(args.raw)
    preds  = load_preds(args.preds)

    common_ids = sorted(set(labels.keys()) & set(preds.keys()))
    y_true_all = [labels[i] for i in common_ids]
    y_pred_all = [preds[i]  for i in common_ids]

    invalid_cnt  = sum(p is None for p in y_pred_all)
    invalid_rate = invalid_cnt / len(common_ids) if common_ids else 0.0
    # To compute ACC, fill None with a fixed label (this will not affect the "incorrect" judgment)
    y_pred_fill = [p if p is not None else "A" for p in y_pred_all]
    acc_all  = accuracy_score(y_true_all, y_pred_fill)

    # F1/confusion matrix on the valid subset
    valid_mask = [p is not None for p in y_pred_all]
    y_true_v = [t for t, m in zip(y_true_all, valid_mask) if m]
    y_pred_v = [p for p, m in zip(y_pred_all, valid_mask) if m]
    micro_f1 = f1_score(y_true_v, y_pred_v, average="micro") if y_true_v else 0.0
    macro_f1 = f1_score(y_true_v, y_pred_v, average="macro") if y_true_v else 0.0
    cm = confusion_matrix(y_true_v, y_pred_v, labels=list(LETTER)) if y_true_v else [[0]*4]*4
    report = classification_report(y_true_v, y_pred_v, labels=list(LETTER), digits=4, zero_division=0) if y_true_v else "N/A"

    from collections import Counter
    cnt_pred = Counter(y_pred_v)
    cnt_true = Counter(y_true_v)

    print(f"Total pairs = {len(common_ids)}")
    print(f"Invalid preds = {invalid_cnt} ({invalid_rate:.2%})  ->  ACC(all) = {acc_all:.4f}")
    print(f"Valid-only micro-F1 = {micro_f1:.4f}  macro-F1 = {macro_f1:.4f}\n")
    print("Pred dist (valid only):", dict(cnt_pred))
    print("True dist (valid only):", dict(cnt_true))
    print("\nConfusion Matrix on valid (rows=true, cols=pred, A-D):")
    print(cm)
    print("\nPer-class report on valid:")
    print(report)

    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({
                "total_pairs": len(common_ids),
                "invalid_preds": invalid_cnt,
                "invalid_rate": invalid_rate,
                "acc_all": acc_all,
                "micro_f1_valid": micro_f1,
                "macro_f1_valid": macro_f1,
                "pred_dist_valid": dict(cnt_pred),
                "true_dist_valid": dict(cnt_true),
                "labels": LETTER,
                "confusion_matrix_valid": cm.tolist()
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to {args.report}")

if __name__ == "__main__":
    main()