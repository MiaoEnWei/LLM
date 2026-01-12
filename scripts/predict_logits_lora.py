# scripts/predict_logits_lora.py
# ---------------------------------------------------------
# V3 - General-purpose logits-based multiple-choice scoring script
# - Supports GPT-2 / LLaMA and other Causal LMs
# - Supports PEFT adapters (LoRA / Prompt Tuning)
# - Uses fp16 by default with full-precision model; optional --use_4bit for large models
# - Input: each line {"id": ..., "prompt": "...Answer (A, B, C, or D):"}
# - Output: each line {"id": ..., "pred_letter": "A/B/C/D", "score": logprob}
# ---------------------------------------------------------

import argparse
import json
import os
import torch
from torch.nn.functional import log_softmax
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

LETTER = "ABCD"


def load_model_and_tokenizer(base_model_path, adapter_path, local_only=True, use_4bit=False):
    """Load the base model + an optional PEFT adapter."""
    if adapter_path and not PEFT_AVAILABLE:
        raise ImportError("Detected --adapter, but the 'peft' library is not installed. Please run: pip install peft")

    print(f"[predict] Loading tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=True,
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Helps truncation + alignment

    print(f"[predict] Loading base model from: {base_model_path}")
    if use_4bit:
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError("bitsandbytes is required to use --use_4bit. Please install it first.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quant_config,
            device_map="auto",
            local_files_only=local_only,
        )
    else:
        # GPT-2 / smaller models can use this branch
        kwargs = {"local_files_only": local_only}
        if torch.cuda.is_available():
            kwargs.update(
                dict(
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **kwargs,
        )

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if adapter_path:
        print(f"[predict] Loading PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=local_only,
        )
        print("[predict] Adapter applied.")

    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_model",
        required=True,
        help="Base model path (e.g., ./gpt2, ./llama2)",
    )
    ap.add_argument(
        "--adapter",
        default="",
        help="PEFT adapter directory (e.g., out_gpt2_official_prompt_tuning)",
    )
    ap.add_argument(
        "--infile",
        required=True,
        help="Input prompts.jsonl (each line contains id / prompt)",
    )
    ap.add_argument(
        "--outfile",
        required=True,
        help="Output preds.jsonl file path",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size (reduce for large models)",
    )
    ap.add_argument(
        "--local_files_only",
        action="store_true",
        help="Force using local files only",
    )
    ap.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization for large models (usually not needed for GPT-2)",
    )

    args = ap.parse_args()

    # You always use local paths, so lock this to True.
    args.local_files_only = True

    if not args.adapter:
        print("[WARN] No --adapter provided; evaluating with the base model only.")

    model, tok = load_model_and_tokenizer(
        args.base_model,
        args.adapter,
        local_only=args.local_files_only,
        use_4bit=args.use_4bit,
    )
    device = model.device

    # Read prompts
    prompts = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            pid = o["id"]
            p = o["prompt"]
            prompts.append((pid, p))

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    w = open(args.outfile, "w", encoding="utf-8")

    # Pre-compute token ids for A/B/C/D
    label_tokens = {}
    for L in LETTER:
        ids = tok.encode(L, add_special_tokens=False)
        if len(ids) == 0:
            raise ValueError(f"Failed to encode letter {L} into a token id")
        label_tokens[L] = ids[0]  # For 'A','B','C','D' this is usually a single token
    print(f"[predict] Label token ids: {label_tokens}")

    total = len(prompts)
    processed = 0

    for i in range(0, total, args.batch_size):
        batch = prompts[i : i + args.batch_size]
        ids = [x[0] for x in batch]
        texts = [x[1] for x in batch]

        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, T, V]
            logp = log_softmax(logits, dim=-1)

        # For each sample, find the last non-pad position, and use that position's
        # distribution to score the next token.
        # We score A/B/C/D using the logprob at that position.
        last_indices = attention_mask.sum(dim=1) - 1  # [B]

        for b_idx, ex_id in enumerate(ids):
            pos = last_indices[b_idx].item()
            # Logprob vector at the selected position
            lp_vec = logp[b_idx, pos]  # [V]

            best_L = None
            best_score = float("-inf")

            for L in LETTER:
                tid = label_tokens[L]
                s = lp_vec[tid].item()
                if s > best_score:
                    best_score = s
                    best_L = L

            w.write(
                json.dumps(
                    {
                        "id": ex_id,
                        "pred_letter": best_L,
                        "score": best_score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        processed += len(batch)
        if processed % (args.batch_size * 10) == 0 or processed == total:
            print(f"[predict] Processed {processed} / {total}")

    w.close()
    print(f"[predict] Saved preds to: {args.outfile}  total={total}")


if __name__ == "__main__":
    main()
