"""
PubMedQA 长答案生成评测 (Run-wise Saving) - BERTScore + (可选)决策ACC + 4bit/8bit量化
- 每次实验创建独立 run 目录，避免覆盖
- 保存：summary.json / all_results.jsonl / case reports
- 支持：Base vs Adapter（PEFT）
- 支持：decision acc（Yes/No/Maybe）可选
- FIX: limit=1000 仍只有 100：不再对全列 dropna，只对必要列 subset dropna，并打印数量
- Metric: BERTScore（默认使用 F1 作为逐样本 score 用于显著性分析）

示例（llama2 4bit）：
python scripts/eval_pubmedqa_gen_v3.py \
  --parquet ./data/pubmedqa_hf/pqa_labeled_splits/test.parquet \
  --model ./llama2 \
  --adapter ./out_llama2_pubmedqa_prompt_tuning_5k_e1 \
  --limit 1000 --max_new_tokens 128 --seed 2025 --with_decision_acc --quiet \
  --load_in_4bit
"""

import os
import re
import json
import random
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
import evaluate

# 检查 PEFT
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def join_context(c, max_chars: int) -> str:
    try:
        if not isinstance(c, dict):
            return ""
        ctxs = (c or {}).get("contexts", [])
        if not ctxs:
            return ""
        return " ".join([str(x) for x in ctxs])[:max_chars]
    except Exception:
        return ""


def build_prompt(tok, use_chat_template: bool, q: str, ctx: str) -> str:
    if use_chat_template and hasattr(tok, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": "You are a helpful biomedical QA assistant."},
            {
                "role": "user",
                "content": (
                    f"Question: {q}\nContext: {ctx}\n"
                    "Provide a concise rationale (1-2 sentences) and end with a final decision word among [Yes, No, Maybe].\n"
                    "Answer:"
                ),
            },
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return TEMPLATE.format(q=q, ctx=ctx)


def extract_decision(text: str):
    t = (text or "").strip().lower()
    m = re.search(r"\b(yes|no|maybe)\b(?=[.?!]?$|\s*$)", t)
    if m:
        return m.group(1)
    cand = re.findall(r"\b(yes|no|maybe)\b", t)
    return cand[-1] if cand else None


def decision_metrics(pred_list, gold_list):
    total = len(gold_list)
    if total == 0:
        return {"acc": 0.0, "support": 0, "format_errors": 0, "confusion": {}}

    preds_filled = [p if p is not None else "invalid" for p in pred_list]
    correct = sum(1 for p, g in zip(preds_filled, gold_list) if p == g)

    labels = ["yes", "no", "maybe"]
    valid_indices = [i for i, p in enumerate(pred_list) if p is not None]
    valid_preds = [pred_list[i] for i in valid_indices]
    valid_golds = [gold_list[i] for i in valid_indices]

    conf = Counter((g, p) for g, p in zip(valid_golds, valid_preds))
    confusion = {f"{g}->{p}": conf[(g, p)] for g in labels for p in labels}

    format_error_count = total - len(valid_indices)
    return {
        "acc": correct / total,
        "support": len(valid_indices),
        "format_errors": format_error_count,
        "confusion": confusion,
    }


def normalize_decision(x: str) -> str:
    s = (str(x) or "").strip().lower()
    if s.startswith("y"):
        return "yes"
    if s.startswith("n"):
        return "no"
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
        "Both_Low": sorted(indices, key=lambda i: sums[i])[:top_k],
    }


def save_case_report(
    case_dict,
    questions,
    refs,
    preds_base,
    preds_pt,
    score_base,
    score_pt,
    output_dir="eval_out",
):
    os.makedirs(output_dir, exist_ok=True)

    for label, ids in case_dict.items():
        if not ids:
            continue

        filename = f"report_{label}.txt"
        path = os.path.join(output_dir, filename)
        print(f">>> Saving report for [{label}] ({len(ids)} samples) -> {path}")

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"=== {label} (Top {len(ids)}) ===\n")
            f.write("Metric: BERTScore-F1\n")
            f.write("Format: [Mark | Score]: Output\n\n")

            for idx in ids:
                s_base = score_base[idx]
                s_pt = score_pt[idx]
                diff = s_pt - s_base

                if "Most_Improved" in label:
                    tag_base = f"❌ Base (Low | {s_base:.4f})"
                    tag_pt = f"✅ PT   (High| {s_pt:.4f})"
                elif "Most_Degraded" in label:
                    tag_base = f"✅ Base (High| {s_base:.4f})"
                    tag_pt = f"❌ PT   (Low | {s_pt:.4f})"
                elif "Both_High" in label:
                    tag_base = f"✅ Base (High| {s_base:.4f})"
                    tag_pt = f"✅ PT   (High| {s_pt:.4f})"
                else:
                    tag_base = f"❌ Base (Low | {s_base:.4f})"
                    tag_pt = f"❌ PT   (Low | {s_pt:.4f})"

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

    # 用 --percentile 当 top_k（沿用你原逻辑）
    ap.add_argument("--percentile", type=int, default=5, help="Top K significant samples")

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_ctx_chars", type=int, default=4000, help="Trim context by chars before tokenization")
    ap.add_argument("--max_input_tokens", type=int, default=2048, help="Tokenizer max_length for inputs (tokens)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--with_decision_acc", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="Silence logs")

    # --- BERTScore 参数 ---
    ap.add_argument(
        "--bertscore_model_type",
        default="allenai/scibert_scivocab_uncased",
        help="BERTScore backbone (biomed推荐SciBERT；也可换 PubMedBERT 等)",
    )
    ap.add_argument("--bertscore_batch_size", type=int, default=16, help="BERTScore compute batch size")
    ap.add_argument("--bertscore_rescale", action="store_true", help="BERTScore rescale_with_baseline")

    # --- 量化参数（llama2 建议 4bit）---
    ap.add_argument("--load_in_8bit", action="store_true", help="bitsandbytes 8-bit quantization")
    ap.add_argument("--load_in_4bit", action="store_true", help="bitsandbytes 4-bit quantization (nf4)")

    return ap.parse_args()


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, tok, questions, ctx_list, args, label="Model", batch_size=4):
    preds = []
    print(f"\nStarting Inference for {label} (Batch Size={batch_size})...")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    all_prompts = [build_prompt(tok, args.use_chat_template, q, ctx) for q, ctx in zip(questions, ctx_list)]
    dev = get_model_device(model)

    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i : i + batch_size]

        inputs = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = out[:, input_len:]
        batch_preds = tok.batch_decode(generated_ids, skip_special_tokens=True)
        preds.extend([p.strip() for p in batch_preds])

        if (i + batch_size) % 50 == 0:
            print(f"  Processed {min(i + batch_size, len(questions))}/{len(questions)}")

    return preds


def compute_bertscore(preds, refs, args):
    """
    使用 evaluate 的 bertscore metric。
    注意：需要 pip install bert_score
    """
    bertscore = evaluate.load("bertscore")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    res = bertscore.compute(
        predictions=preds,
        references=refs,
        lang="en",
        model_type=args.bertscore_model_type,
        device=dev,
        batch_size=args.bertscore_batch_size,
        rescale_with_baseline=bool(args.bertscore_rescale),
    )

    p_list = [float(x) for x in res["precision"]]
    r_list = [float(x) for x in res["recall"]]
    f1_list = [float(x) for x in res["f1"]]

    agg = {
        "precision_mean": float(np.mean(p_list)) if p_list else 0.0,
        "recall_mean": float(np.mean(r_list)) if r_list else 0.0,
        "f1_mean": float(np.mean(f1_list)) if f1_list else 0.0,
        "model_type": args.bertscore_model_type,
        "rescale_with_baseline": bool(args.bertscore_rescale),
        "device": dev,
        "batch_size": args.bertscore_batch_size,
        "hashcode": res.get("hashcode"),
    }
    per = {"precision": p_list, "recall": r_list, "f1": f1_list}
    return agg, per


def main():
    args = build_args()
    if args.quiet:
        hf_logging.set_verbosity_error()
    set_seed(args.seed)

    # ---------- run 目录 ----------
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = os.path.basename(args.model.rstrip("/"))
    adapter_tag = os.path.basename(args.adapter.rstrip("/")) if args.adapter else "no_adapter"
    qtag = "4bit" if args.load_in_4bit else ("8bit" if args.load_in_8bit else "fp16")
    run_name = f"run_{time_str}_{model_tag}_{adapter_tag}_{qtag}_limit{args.limit}_seed{args.seed}"
    base_out_dir = "eval_out"
    run_dir = os.path.join(base_out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n>>> Current run dir: {run_dir}\n")
    # -----------------------------

    print(f"Reading data from {args.parquet}")
    tbl = pq.read_table(args.parquet)
    df_all = tbl.to_pandas()

    # ===== FIX: 只对必要列 dropna =====
    need_cols = ["question", "long_answer"]
    if args.with_decision_acc and "final_decision" in df_all.columns:
        need_cols.append("final_decision")

    before_n = len(df_all)
    df = df_all.dropna(subset=need_cols).head(args.limit)
    after_n = len(df)
    print(f"[Data] total_rows={before_n}, after_dropna(subset={need_cols})={after_n}, args.limit={args.limit}")
    if after_n < args.limit:
        print(f"[Warn] 过滤后样本数({after_n})小于 limit({args.limit})，说明数据本身可用行不足或 subset 过严。")
    # =================================

    questions = df["question"].astype(str).tolist()
    if "context" in df.columns:
        ctx_list = df["context"].map(lambda c: join_context(c, args.max_ctx_chars)).tolist()
    else:
        ctx_list = [""] * len(df)
    refs = df["long_answer"].astype(str).tolist()

    # load tokenizer
    print(f"Loading Tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # load base model (支持 4bit/8bit)
    print(f"Loading Base Model: {args.model} ({qtag}, device_map=auto)")
    loader_kwargs = dict(
        device_map="auto",
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("不能同时开启 --load_in_4bit 和 --load_in_8bit")

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        loader_kwargs["quantization_config"] = bnb_config
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        loader_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        loader_kwargs["torch_dtype"] = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(args.model, **loader_kwargs)
    base_model.config.pad_token_id = tok.pad_token_id
    base_model.eval()

    preds_base = run_inference(base_model, tok, questions, ctx_list, args, label="Base Model")

    if args.adapter:
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is not installed but --adapter was provided. pip install peft")
        print(f"\nLoading Adapter from {args.adapter}...")
        pt_model = PeftModel.from_pretrained(base_model, args.adapter, local_files_only=args.local_files_only)
        pt_model.eval()
        preds_pt = run_inference(pt_model, tok, questions, ctx_list, args, label="PT Model")
    else:
        print("\nNo adapter provided. Skipping PT inference.")
        preds_pt = preds_base[:]

    # -------------------- BERTScore --------------------
    print("\nCalculating BERTScore...")
    bs_base_agg, bs_base_per = compute_bertscore(preds_base, refs, args)
    bs_pt_agg, bs_pt_per = compute_bertscore(preds_pt, refs, args)

    print(f"\n=== Base BERTScore (mean) ===\n{json.dumps(bs_base_agg, indent=2, ensure_ascii=False)}")
    print(f"\n=== PT   BERTScore (mean) ===\n{json.dumps(bs_pt_agg, indent=2, ensure_ascii=False)}")

    scores_base = bs_base_per["f1"]
    scores_pt = bs_pt_per["f1"]
    # --------------------------------------------------

    k = args.percentile if args.limit >= args.percentile else max(1, args.limit // 2)
    print(f"\nAnalyzing cases (Top {k} significant samples)...")
    case_indices = find_significant_cases(scores_base, scores_pt, top_k=k)
    save_case_report(
        case_indices,
        questions,
        refs,
        preds_base,
        preds_pt,
        scores_base,
        scores_pt,
        output_dir=run_dir,
    )

    # decision acc（可选）
    decision_summary_base = None
    decision_summary_pt = None
    if args.with_decision_acc and "final_decision" in df.columns:
        gold = [normalize_decision(x) for x in df["final_decision"]]
        decision_summary_base = decision_metrics([extract_decision(x) for x in preds_base], gold)
        decision_summary_pt = decision_metrics([extract_decision(x) for x in preds_pt], gold)
        print(f"\n=== Base ACC ===\n{json.dumps(decision_summary_base, indent=2, ensure_ascii=False)}")
        print(f"\n=== PT   ACC ===\n{json.dumps(decision_summary_pt, indent=2, ensure_ascii=False)}")

    # summary.json
    summary = {
        "args": vars(args),
        "run_name": run_name,
        "n_samples": len(df),
        "base_model": args.model,
        "adapter": args.adapter,
        "quant": qtag,
        "bertscore_base_agg": bs_base_agg,
        "bertscore_pt_agg": bs_pt_agg,
        "decision_base": decision_summary_base,
        "decision_pt": decision_summary_pt,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # all_results.jsonl
    out_file = os.path.join(run_dir, "all_results.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(len(df)):
            rec = {
                "id": int(i),
                "question": questions[i],
                "context": ctx_list[i],
                "ref": refs[i],
                "pred_base": preds_base[i],
                "pred_pt": preds_pt[i],
                "bertscore_precision_base": float(bs_base_per["precision"][i]),
                "bertscore_recall_base": float(bs_base_per["recall"][i]),
                "bertscore_f1_base": float(bs_base_per["f1"][i]),
                "bertscore_precision_pt": float(bs_pt_per["precision"][i]),
                "bertscore_recall_pt": float(bs_pt_per["recall"][i]),
                "bertscore_f1_pt": float(bs_pt_per["f1"][i]),
            }
            if args.with_decision_acc and "final_decision" in df.columns:
                rec["gold_decision"] = normalize_decision(df["final_decision"].iloc[i])
                rec["pred_decision_base"] = extract_decision(preds_base[i])
                rec["pred_decision_pt"] = extract_decision(preds_pt[i])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved detailed per-sample results to {out_file}\n")


if __name__ == "__main__":
    main()
