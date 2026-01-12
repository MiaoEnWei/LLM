"""
PubMedQA decision consistency evaluation (Decision-only ACC)

- Read question / context / final_decision from the official parquet
- Use a Causal LM (GPT-2 / LLaMA, etc.) to generate [only Yes/No/Maybe]
- Compute Yes/No/Maybe ACC + confusion matrix
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

# Optional: PEFT adapter support (Prompt Tuning / LoRA)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def set_seed(seed: int):
    """Make results as reproducible as possible (CPU/GPU)."""
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
    """Flatten the context structure into a string; use only the contexts field and truncate."""
    try:
        ctxs = (c or {}).get("contexts", [])
        if not ctxs:
            return ""
        out = " ".join(ctxs)
        return out[:max_chars]
    except Exception:
        return ""


def build_prompt(tok, use_chat_template: bool, q: str, ctx: str) -> str:
    """
    Decision-only prompt: require the model to output only Yes/No/Maybe
    """
    if use_chat_template and hasattr(tok, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": "You are a biomedical QA assistant."},
            {
                "role": "user",
                "content": (
                    f"Question: {q}\n"
                    f"Context: {ctx}\n\n"
                    "Answer with only one word among [Yes, No, Maybe].\n"
                    "Answer:"
                ),
            },
        ]
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    # Non-chat model: use a plain prompt
    return (
        "You are a biomedical QA assistant.\n"
        f"Question: {q}\n"
        f"Context: {ctx}\n\n"
        "Answer with only one word among [Yes, No, Maybe].\n"
        "Answer:"
    )


def extract_decision(text: str):
    """
    Extract yes/no/maybe from generated text (case-insensitive):
    1) Try near the beginning (matches the expectation of "only one word")
    2) Fallback: take the last occurrence in the full text
    """
    t = (text or "").strip().lower()

    # Strip surrounding quotes/punctuation
    t = t.strip(' "\'\n\t.')

    # Try to match "the whole output is a single word"
    m = re.match(r"^(yes|no|maybe)\b\.?$", t)
    if m:
        return m.group(1)

    # Find all yes/no/maybe and take the last one
    cand = re.findall(r"\b(yes|no|maybe)\b", t)
    return cand[-1] if cand else None


def normalize_decision(x: str) -> str:
    """Normalize gold decision to three lowercase classes: yes/no/maybe."""
    s = (str(x) or "").strip().lower()
    if s.startswith("y"):
        return "yes"
    if s.startswith("n"):
        return "no"
    return "maybe"


def decision_metrics(pred_list, gold_list):
    """Compute basic ACC and confusion stats."""
    assert len(pred_list) == len(gold_list)
    valid_idx = [i for i, p in enumerate(pred_list) if p is not None]
    if not valid_idx:
        return {"acc": 0.0, "support": 0, "coverage": 0.0, "confusion": {}}

    preds = [pred_list[i] for i in valid_idx]
    golds = [gold_list[i] for i in valid_idx]

    correct = sum(int(p == g) for p, g in zip(preds, golds))
    acc = correct / len(valid_idx)
    coverage = len(valid_idx) / len(pred_list)

    labels = ["yes", "no", "maybe"]
    conf = Counter((g, p) for g, p in zip(golds, preds))
    confusion = {f"{g}->{p}": conf[(g, p)] for g in labels for p in labels}

    return {
        "acc": acc,
        "support": len(valid_idx),
        "coverage": coverage,
        "confusion": confusion,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Path to pqa_labeled_test.parquet")
    ap.add_argument("--model", required=True, help="HF model name or local weights directory")
    ap.add_argument("--adapter", default="", help="PEFT adapter directory (optional: Prompt Tuning / LoRA)")
    ap.add_argument("--use_chat_template", action="store_true", help="Enable chat template for -chat models")
    ap.add_argument("--limit", type=int, default=500, help="Number of samples to evaluate (take from the head)")
    ap.add_argument("--max_new_tokens", type=int, default=4, help="Max tokens to generate (decision task needs few)")
    ap.add_argument("--max_ctx_chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true", help="Silence Transformers logs")
    ap.add_argument("--local_files_only", action="store_true", help="Use local model files only")
    args = ap.parse_args()

    if args.quiet:
        hf_logging.set_verbosity_error()

    set_seed(args.seed)

    # Read & align data
    tbl = pq.read_table(args.parquet)
    df_full = tbl.to_pandas()

    needed_cols = ["question", "context", "final_decision"]
    for col in needed_cols:
        if col not in df_full.columns:
            raise ValueError(f"Missing column in parquet: {col}. Current columns: {list(df_full.columns)}")

    df = df_full[needed_cols].dropna().head(args.limit)
    df["ctx"] = df["context"].map(lambda c: join_context(c, args.max_ctx_chars))

    questions = df["question"].tolist()
    ctx_list = df["ctx"].tolist()
    gold_decisions = [normalize_decision(x) for x in df["final_decision"].tolist()]

    # Load model & tokenizer
    use_fp16 = torch.cuda.is_available()
    print(f"[DecisionEval] Loading tokenizer from: {args.model}")
    tok = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[DecisionEval] Loading base model from: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=(torch.float16 if use_fp16 else torch.float32),
        device_map="auto" if use_fp16 else None,
        local_files_only=args.local_files_only,
    )
    model.config.pad_token_id = tok.pad_token_id

    # Optionally load PEFT adapter
    if args.adapter:
        if not PEFT_AVAILABLE:
            raise ImportError("You specified --adapter, but peft is not installed. Please run: pip install peft")
        print(f"[DecisionEval] Loading PEFT adapter from: {args.adapter}")
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            local_files_only=args.local_files_only,
        )
    model.eval()

    preds = []

    for i, (q, ctx) in enumerate(zip(questions, ctx_list)):
        prompt = build_prompt(tok, args.use_chat_template, q, ctx)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # use greedy decoding first for decision tasks
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen = tok.decode(gen_ids, skip_special_tokens=True).strip()
        preds.append(gen)

        if (i + 1) % 20 == 0:
            print(f"[DecisionEval] {i+1}/{len(df)}")

    # Extract decisions & compute metrics
    pred_decisions = [extract_decision(t) for t in preds]
    metrics = decision_metrics(pred_decisions, gold_decisions)

    print("\n=== PubMedQA Decision-only Evaluation (Yes/No/Maybe) ===")
    print(json.dumps(
        {
            "acc": round(metrics["acc"], 6),
            "support_extracted": metrics["support"],
            "coverage": round(metrics["coverage"], 6),
            "confusion": metrics["confusion"],
        },
        indent=2,
        ensure_ascii=False,
    ))

    # Optional: save raw predictions for debugging
    os.makedirs("eval_out", exist_ok=True)
    out_path = "eval_out/pubmedqa_decision_only_samples.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(df)):
            f.write(json.dumps(
                {
                    "idx": i,
                    "question": questions[i],
                    "ctx": ctx_list[i],
                    "gold_final": gold_decisions[i],
                    "gen_text": preds[i],
                    "pred_decision": pred_decisions[i],
                },
                ensure_ascii=False,
            ) + "\n")
    print(f"\nSaved samples -> {out_path}")


if __name__ == "__main__":
    main()
