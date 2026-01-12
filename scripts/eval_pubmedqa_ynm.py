import argparse, os, math, json, re
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import Dataset

LABELS = ["yes", "no", "maybe"]
# Multiple verbalizers to improve robustness (case variants, with period)
VERBALIZERS = {
    "yes":   ["Yes", "yes", "YES", "Yes.", "yes."],
    "no":    ["No", "no", "NO", "No.", "no."],
    "maybe": ["Maybe", "maybe", "MAYBE", "Maybe.", "maybe."],
}

TEMPLATE = (
    "You are a biomedical QA assistant.\n"
    "Question: {q}\n"
    "Context: {ctx}\n\n"
    "Answer with exactly one word from [Yes, No, Maybe].\n"
    "Final answer: "
)

def join_context_field(ctx_obj, max_chars=4000):
    # ctx_obj: dict, with "contexts": list[str]
    if isinstance(ctx_obj, dict) and "contexts" in ctx_obj and ctx_obj["contexts"]:
        txt = " ".join(ctx_obj["contexts"])
        return txt[:max_chars]
    return ""

@torch.no_grad()
def score_candidate(model, tokenizer, prompt_ids, cand_text, device):
    """
    Compute the log-likelihood log P(cand | prompt). Approach: concatenate prompt + cand,
    and only sum the log-probabilities for tokens belonging to cand.
    """
    # Note: prepend a space to cand_text for more stable English tokenization
    cand = " " + cand_text
    cand_ids = tokenizer.encode(cand, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids + cand_ids], device=device)
    outputs = model(input_ids, labels=input_ids)
    # With labels aligned to input_ids, loss is averaged; we instead compute token-wise logprob ourselves.
    # We take the cross-entropy for the last len(cand_ids) tokens:
    # cross-entropy = -logprob
    # Extract probabilities from logits:
    logits = outputs.logits[:, :-1, :]  # predict next token
    shift_labels = input_ids[:, 1:]
    # Keep only the candidate segment
    keep = shift_labels.shape[1] - len(cand_ids)
    cand_logits = logits[:, keep:, :]
    cand_labels = shift_labels[:, keep:]
    log_probs = torch.log_softmax(cand_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, cand_labels.unsqueeze(-1)).squeeze(-1)  # [1, Lc]
    return token_logprobs.sum().item()  # scalar

def pick_label_by_ll(model, tokenizer, prompt, device):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # To avoid an overly long context, truncate to the model max input (keep the tail)
    max_len = getattr(model.config, "max_position_embeddings", 2048) - 32
    if len(prompt_ids) > max_len:
        prompt_ids = prompt_ids[-max_len:]
    scores = {}
    for lab in LABELS:
        cand_list = VERBALIZERS[lab]
        # For multiple verbalizers of the same label, use logsumexp to ensemble “forms”
        cand_scores = []
        for cand in cand_list:
            s = score_candidate(model, tokenizer, prompt_ids, cand, device)
            cand_scores.append(s)
        # logsumexp for stable aggregation
        m = max(cand_scores)
        scores[lab] = m + math.log(sum(math.exp(x - m) for x in cand_scores))
    # Pick the label with the highest score
    pred = max(scores.items(), key=lambda x: x[1])[0]
    return pred, scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="pqa_labeled/train-00000-of-00001.parquet")
    ap.add_argument("--model", required=True, help="HF model name or local path, e.g., gpt2 / ./gpt2 / meta-llama/Llama-2-7b-hf (local)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--max_ctx_chars", type=int, default=3000)
    ap.add_argument("--use_chat_template", action="store_true",
                    help="Enable chat template for models that require it, such as Llama-2-chat")
    args = ap.parse_args()

    # Read local labeled parquet
    table = pq.read_table(args.parquet)
    df = table.to_pandas()
    df = df[["question", "context", "final_decision"]].dropna()
    df = df.head(args.limit).copy()
    df["context_text"] = df["context"].map(lambda c: join_context_field(c, args.max_ctx_chars))

    # Load model
    print(f"Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        device_map="auto" if args.device.startswith("cuda") else None
    )
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    y_true, y_pred = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        q, ctx, y = row.question, row.context_text, str(row.final_decision).lower().strip()
        if args.use_chat_template and hasattr(tok, "apply_chat_template"):
            msgs = [
                {"role": "system", "content": "You are a helpful biomedical QA assistant."},
                {"role": "user", "content":
                 f"Question: {q}\nContext: {ctx}\nAnswer with one of [Yes, No, Maybe].\nFinal answer: "}
            ]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = TEMPLATE.format(q=q, ctx=ctx)

        pred, _ = pick_label_by_ll(model, tok, prompt, device)
        y_true.append(y if y in LABELS else "maybe")  # Safety: map unexpected values to maybe
        y_pred.append(pred)

        if (i + 1) % 50 == 0:
            acc = accuracy_score(y_true, y_pred)
            print(f"[{i+1}/{len(df)}] running acc={acc:.4f}")

    acc = accuracy_score(y_true, y_pred)
    print("\n== Results ==")
    print("ACC:", f"{acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))

    # Save predictions
    out = [{"question": df.iloc[i]["question"],
            "pred": y_pred[i], "gold": y_true[i]} for i in range(len(y_true))]
    os.makedirs("eval_out", exist_ok=True)
    with open("eval_out/pqa_labeled_eval.jsonl", "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("Saved -> eval_out/pqa_labeled_eval.jsonl")

if __name__ == "__main__":
    main()