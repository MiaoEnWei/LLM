"""
PubMedQA long-answer generation evaluation v4 (Final: Scores in Labels)
- Core improvement: in the generated report, display ROUGE scores directly in the model output labels for more intuitive comparison.
- Logic preserved: delta ranking (Delta Ranking), ensuring cases will not be empty.
"""

import os
import re
import json
import random
import argparse
from collections import Counter

import numpy as np
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
import evaluate

# Check PEFT
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

TEMPLATE = (
    "You are a biomedical QA assistant.\n"
    "Question: {q}\n"
    "Context: {ctx}\n\n"
    "Provide a concise rationale (1-2 sentences) and end with a final decision word among [Yes, No, Maybe].\n"
    "Answer: "
)

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.use_deterministic_algorithms(True) # Disabled to prevent CuBLAS error
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def join_context(c, max_chars: int) -> str:
    try:
        ctxs = (c or {}).get("contexts", [])
        if not ctxs: return ""
        return " ".join(ctxs)[:max_chars]
    except Exception:
        return ""

def build_prompt(tok, use_chat_template: bool, q: str, ctx: str) -> str:
    if use_chat_template and hasattr(tok, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": "You are a helpful biomedical QA assistant."},
            {"role": "user", "content": f"Question: {q}\nContext: {ctx}\nProvide a concise rationale (1-2 sentences) and end with a final decision word among [Yes, No, Maybe].\nAnswer:"},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return TEMPLATE.format(q=q, ctx=ctx)

def extract_decision(text: str):
    t = (text or "").strip().lower()
    m = re.search(r"\b(yes|no|maybe)\b(?=[.?!]?$|\s*$)", t)
    if m: return m.group(1)
    cand = re.findall(r"\b(yes|no|maybe)\b", t)
    return cand[-1] if cand else None

def decision_metrics(pred_list, gold_list):
    valid_idx = [i for i, p in enumerate(pred_list) if p is not None]
    if not valid_idx: return {"acc": 0.0, "support": 0, "confusion": {}}
    preds = [pred_list[i] for i in valid_idx]
    golds = [gold_list[i] for i in valid_idx]
    correct = sum(int(p == g) for p, g in zip(preds, golds))
    labels = ["yes", "no", "maybe"]
    conf = Counter((g, p) for g, p in zip(golds, preds))
    confusion = {f"{g}->{p}": conf[(g, p)] for g in labels for p in labels}
    return {"acc": correct / len(valid_idx), "support": len(valid_idx), "confusion": confusion}

def normalize_decision(x: str) -> str:
    s = (str(x) or "").strip().lower()
    if s.startswith("y"): return "yes"
    if s.startswith("n"): return "no"
    return "maybe"

def find_significant_cases(base_scores, pt_scores, top_k=5):
    n = len(base_scores)
    diffs = [pt - base for pt, base in zip(pt_scores, base_scores)]
    sums = [pt + base for pt, base in zip(pt_scores, base_scores)]
    indices = list(range(n))
    
    return {
        "Most_Improved_BaseLow_PTHigh": sorted(indices, key=lambda i: diffs[i], reverse=True)[:top_k],
        "Most_Degraded_BaseHigh_PTLow": sorted(indices, key=lambda i: diffs[i])[:top_k],
        "Both_High": sorted(indices, key=lambda i: sums[i], reverse=True)[:top_k],
        "Both_Low": sorted(indices, key=lambda i: sums[i])[:top_k]
    }

# ==================================================================
# [Change] Write the exact scores into the [Label]
# ==================================================================
def save_case_report(case_dict, questions, ctx_list, refs, preds_base, preds_pt, 
                     score_base, score_pt, output_dir="eval_out"):
    os.makedirs(output_dir, exist_ok=True)
    
    for label, ids in case_dict.items():
        if not ids: continue
        
        filename = f"report_{label}.txt"
        path = os.path.join(output_dir, filename)
        print(f">>> Saving report for [{label}] ({len(ids)} samples) -> {path}")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"=== {label} (Top {len(ids)}) ===\n")
            f.write("Format: [Mark | Score]: Output\n\n")
            
            for idx in ids:
                s_base = score_base[idx]
                s_pt = score_pt[idx]
                diff = s_pt - s_base
                
                # Dynamically generate labels with scores
                if "Most_Improved" in label:
                    tag_base = f"Base (Low | {s_base:.4f})"
                    tag_pt   = f"PT   (High| {s_pt:.4f})"
                elif "Most_Degraded" in label:
                    tag_base = f"Base (High| {s_base:.4f})"
                    tag_pt   = f"PT   (Low | {s_pt:.4f})"
                elif "Both_High" in label:
                    tag_base = f"Base (High| {s_base:.4f})"
                    tag_pt   = f"PT   (High| {s_pt:.4f})"
                else: # Both_Low
                    tag_base = f"Base (Low | {s_base:.4f})"
                    tag_pt   = f"PT   (Low | {s_pt:.4f})"

                f.write(f"Sample ID : {idx}\n")
                f.write(f"Question  : {questions[idx]}\n")
                f.write(f"Diff      : {diff:+.4f}\n")
                f.write("-" * 20 + " Model Outputs " + "-" * 20 + "\n")
                f.write(f"[{tag_base}]: {preds_base[idx]}\n")
                f.write(f"[{tag_pt}  ]: {preds_pt[idx]}\n")
                f.write("-" * 20 + " Reference " + "-" * 24 + "\n")
                f.write(f"[Reference]           : {refs[idx]}\n")
                f.write("=" * 60 + "\n\n")

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="pqa_labeled parquet path")
    ap.add_argument("--model", required=True, help="Base model name or path")
    ap.add_argument("--adapter", default="", help="PEFT adapter path (optional)")
    ap.add_argument("--limit", type=int, default=200, help="Number of samples to eval")
    
    # Use --percentile consistently
    ap.add_argument("--percentile", type=int, default=5, help="Top K significant samples")
    
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_ctx_chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--with_decision_acc", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="Silence logs")
    return ap.parse_args()

def run_inference(model, tok, questions, ctx_list, args, label="Model"):
    preds = []
    print(f"\nStarting Inference for {label}...")
    for i, (q, ctx) in enumerate(zip(questions, ctx_list)):
        prompt = build_prompt(tok, args.use_chat_template, q, ctx)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        preds.append(tok.decode(gen_ids, skip_special_tokens=True).strip())
        if (i+1) % 50 == 0: print(f"  Processed {i+1}/{len(questions)}")
    return preds

def main():
    args = build_args()
    hf_logging.set_verbosity_error()
    set_seed(args.seed)

    print(f"Reading data from {args.parquet}")
    tbl = pq.read_table(args.parquet)
    df = tbl.to_pandas().dropna().head(args.limit)
    questions = df["question"].tolist()
    ctx_list = df["context"].map(lambda c: join_context(c, args.max_ctx_chars)).tolist()
    refs = df["long_answer"].tolist()

    use_fp16 = torch.cuda.is_available()
    print(f"Loading Base Model: {args.model} (Forcing FP32 for Stability)")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=args.local_files_only)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,  # <--- force float32
        device_map="auto",    # if VRAM is enough use auto; otherwise remove this line to default to GPU0
        local_files_only=args.local_files_only
    )
    base_model.config.pad_token_id = tok.pad_token_id
    base_model.eval()

    preds_base = run_inference(base_model, tok, questions, ctx_list, args, label="Base Model")

    if args.adapter:
        print(f"\nLoading Adapter from {args.adapter}...")
        pt_model = PeftModel.from_pretrained(base_model, args.adapter, local_files_only=args.local_files_only)
        pt_model.eval()
        preds_pt = run_inference(pt_model, tok, questions, ctx_list, args, label="PT Model")
    else:
        print("\nNo adapter provided. Skipping PT inference.")
        preds_pt = preds_base[:]

    print("\nCalculating ROUGE scores...")
    rouge = evaluate.load("rouge")
    r_base_agg = rouge.compute(predictions=preds_base, references=refs)
    r_pt_agg = rouge.compute(predictions=preds_pt, references=refs)
    print(f"\n=== Base ROUGE ===\n{json.dumps(r_base_agg, indent=2)}")
    print(f"\n=== PT   ROUGE ===\n{json.dumps(r_pt_agg, indent=2)}")

    scores_base = [float(x) for x in rouge.compute(predictions=preds_base, references=refs, use_aggregator=False)["rougeLsum"]]
    scores_pt = [float(x) for x in rouge.compute(predictions=preds_pt, references=refs, use_aggregator=False)["rougeLsum"]]

    # Use args.percentile
    k = args.percentile if args.limit >= args.percentile else max(1, args.limit // 2)
    print(f"\nAnalyzing cases (Top {k} significant samples)...")
    case_indices = find_significant_cases(scores_base, scores_pt, top_k=k)
    save_case_report(case_indices, questions, ctx_list, refs, preds_base, preds_pt, scores_base, scores_pt, output_dir="eval_out")

    if args.with_decision_acc and "final_decision" in df.columns:
        gold = [normalize_decision(x) for x in df["final_decision"]]
        print(f"\n=== Base ACC ===\n{json.dumps(decision_metrics([extract_decision(x) for x in preds_base], gold), indent=2)}")
        print(f"\n=== PT   ACC ===\n{json.dumps(decision_metrics([extract_decision(x) for x in preds_pt], gold), indent=2)}")

    out_file = "eval_out/all_results.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(len(df)):
            f.write(json.dumps({"id": i, "question": questions[i], "ref": refs[i], "pred_base": preds_base[i], "pred_pt": preds_pt[i], "rouge_base": scores_base[i], "rouge_pt": scores_pt[i]}, ensure_ascii=False) + "\n")
    print(f"\nSaved details to {out_file}")

if __name__ == "__main__":
    main()
