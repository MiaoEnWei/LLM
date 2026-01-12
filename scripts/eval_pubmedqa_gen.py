# scripts/eval_pubmedqa_gen.py
"""
PubMedQA long-answer generation evaluation (ROUGE + optional decision consistency ACC)
- Read pqa_labeled from a local Parquet file
- Generative evaluation: use a specified Causal LM (GPT-2 / Llama-2 / -chat)
- Optionally extract Yes/No/Maybe from generated text to compute decision consistency
- Support setting a random seed to improve reproducibility
- Support loading a PEFT adapter (--adapter), e.g., Prompt Tuning / LoRA
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

# Try importing peft (for loading Prompt Tuning / LoRA adapters)
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
    """Try to make runs reproducible (CPU/GPU)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def join_context(c, max_chars: int) -> str:
    """Flatten the context structure into a string; only take the `contexts` field and truncate."""
    try:
        ctxs = (c or {}).get("contexts", [])
        if not ctxs:
            return ""
        out = " ".join(ctxs)
        return out[:max_chars]
    except Exception:
        return ""


def build_prompt(tok, use_chat_template: bool, q: str, ctx: str) -> str:
    """Construct a prompt depending on whether this is a chat model."""
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
    """
    Try to extract yes/no/maybe from the end of the generated long answer (case-insensitive).
    Prefer matching at the sentence end; if that fails, take the last occurrence in the full text.
    """
    t = (text or "").strip().lower()
    m = re.search(r"\b(yes|no|maybe)\b\.?$", t)
    if m:
        return m.group(1)
    cand = re.findall(r"\b(yes|no|maybe)\b", t)
    return cand[-1] if cand else None


def decision_metrics(pred_list, gold_list):
    """Compute basic ACC and confusion statistics (without sklearn)."""
    assert len(pred_list) == len(gold_list)
    valid_idx = [i for i, p in enumerate(pred_list) if p is not None]
    if not valid_idx:
        return {"acc": 0.0, "support": 0, "confusion": {}}

    preds = [pred_list[i] for i in valid_idx]
    golds = [gold_list[i] for i in valid_idx]

    correct = sum(int(p == g) for p, g in zip(preds, golds))
    acc = correct / len(valid_idx)

    labels = ["yes", "no", "maybe"]
    conf = Counter((g, p) for g, p in zip(golds, preds))
    confusion = {f"{g}->{p}": conf[(g, p)] for g in labels for p in labels}

    return {"acc": acc, "support": len(valid_idx), "confusion": confusion}


def one_line(s: str, max_len: int = 300) -> str:
    """When printing examples, compress newlines into one line for easier terminal viewing."""
    if s is None:
        return ""
    return s[:max_len].replace("\n", " ").strip()


def load_model_and_tokenizer(model_path: str, adapter_path: str, local_files_only: bool):
    """Load a base model plus an optional adapter (Prompt Tuning / LoRA) in a unified way."""
    print(f"[eval_pubmedqa] Loading tokenizer from {model_path}")
    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_fp16 = torch.cuda.is_available()
    load_kwargs = {
        "local_files_only": local_files_only,
    }
    if use_fp16:
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    print(f"[eval_pubmedqa] Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs,
    )

    if adapter_path:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "Detected --adapter, but peft is not installed in this environment.\n"
                "Please run: pip install peft"
            )
        print(f"[eval_pubmedqa] Loading PEFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=local_files_only,
        )
        print("[eval_pubmedqa] Adapter applied.")

    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Parquet path for pqa_labeled")
    ap.add_argument("--model", required=True, help="HF model name or local weights directory")
    ap.add_argument("--adapter", default="", help="PEFT adapter directory (Prompt Tuning / LoRA), can be empty")
    ap.add_argument("--use_chat_template", action="store_true", help="Enable chat template for -chat models")
    ap.add_argument("--limit", type=int, default=200, help="Number of evaluation samples (take from head)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_ctx_chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true", help="Silence Transformers logs")
    ap.add_argument("--with_decision_acc", action="store_true", help="Also evaluate decision consistency ACC")
    ap.add_argument("--local_files_only", action="store_true", help="Force using local model/tokenizer files only")
    args = ap.parse_args()

    if args.quiet:
        hf_logging.set_verbosity_error()

    # Random seed
    set_seed(args.seed)

    # Read parquet data
    print(f"[eval_pubmedqa] Loading parquet from {args.parquet}")
    tbl = pq.read_table(args.parquet)
    pdf = tbl.to_pandas()

    # Keep only rows with question / context / long_answer
    pdf = pdf[["question", "context", "long_answer"]].dropna().head(args.limit)
    pdf["ctx"] = pdf["context"].map(lambda c: join_context(c, args.max_ctx_chars))

    # Load model & tokenizer & adapter
    model, tok = load_model_and_tokenizer(
        model_path=args.model,
        adapter_path=args.adapter,
        local_files_only=args.local_files_only,
    )
    device = next(model.parameters()).device
    print(f"[eval_pubmedqa] Model device: {device}")

    preds, refs = [], []

    print(f"[eval_pubmedqa] Start generation, total {len(pdf)} examples...")
    for i, r in pdf.iterrows():
        q = r["question"]
        ctx = r["ctx"]
        ref = r["long_answer"]

        prompt = build_prompt(tok, args.use_chat_template, q, ctx)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Disable sampling for reproducibility
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen = tok.decode(gen_ids, skip_special_tokens=True).strip()

        preds.append(gen)
        refs.append(ref)

        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(pdf)}]")

    # Compute ROUGE
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(predictions=preds, references=refs)
    rouge_res = {k: float(v) for k, v in rouge_res.items()}
    print("\n=== ROUGE (Aggregated) ===")
    print(json.dumps(rouge_res, indent=2, ensure_ascii=False))

    # -------- Print top5 / low5 / median5 (by rougeLsum) --------
    print("\n[eval_pubmedqa] Computing per-example rougeLsum for ranking...")
    rougeLsum_scores = []
    for pred, ref in zip(preds, refs):
        s = rouge.compute(predictions=[pred], references=[ref])["rougeLsum"]
        rougeLsum_scores.append(float(s))

    idx_sorted = sorted(range(len(rougeLsum_scores)), key=lambda i: rougeLsum_scores[i])
    n = len(idx_sorted)

    def show_examples(title, indices):
        print(f"\n[{title}]  (by rougeLsum)")
        for i in indices:
            score = rougeLsum_scores[i]
            q = pdf.iloc[i]["question"]
            print(f"- #{i:04d}  score={score:.4f}")
            print(f"  Q: {one_line(q, 200)}")
            print(f"  PRED: {one_line(preds[i], 300)}")
            print(f"  REF:  {one_line(refs[i], 300)}\n")

    top_k = 5
    low5_idx = idx_sorted[:top_k]
    top5_idx = idx_sorted[-top_k:][::-1]  # High to low
    mid_start = max(0, n // 2 - top_k // 2)
    mid_idx = idx_sorted[mid_start:mid_start + top_k]

    show_examples("TOP 5", top5_idx)
    show_examples("LOW 5", low5_idx)
    show_examples("MEDIAN 5", mid_idx)

    # -------- Decision consistency (Yes / No / Maybe) --------
    if args.with_decision_acc:
        print("\n[eval_pubmedqa] Evaluating decision consistency (Yes/No/Maybe)...")
        gold_df = tbl.to_pandas()[["final_decision"]].dropna().head(args.limit)
        gold = [str(x).lower().strip() for x in gold_df["final_decision"].tolist()]
        pred_dec = [extract_decision(t) for t in preds]
        dec_res = decision_metrics(pred_dec, gold)

        print("\n=== Decision Consistency (Yes/No/Maybe) ===")
        print(
            json.dumps(
                {
                    "acc": round(dec_res["acc"], 6),
                    "support_extracted": dec_res["support"],
                    "confusion": dec_res["confusion"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    # -------- Save samples --------
    os.makedirs("eval_out", exist_ok=True)
    out_path = "eval_out/pqa_labeled_gen_samples.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for q, p, ref in zip(pdf["question"], preds, refs):
            f.write(
                json.dumps({"q": q, "pred": p, "ref": ref}, ensure_ascii=False)
                + "\n"
            )
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
