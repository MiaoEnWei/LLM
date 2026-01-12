# scripts/predict_generate_lora.py
# This is an evaluation script that uses model.generate() (generation-based).
# It is more memory-efficient than the "logits-based" method, and is suitable
# for evaluating Llama-2 on a 12GB GPU.

import argparse, json, os, torch, re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from transformers.utils import logging
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logging.set_verbosity_error()

LETTER = "ABCD"

def extract_letter(txt: str) -> str:
    """Extract the first A, B, C, or D from the generated text."""
    txt = txt.strip()
    # Match a leading A, B, C, or D (case-insensitive)
    m = re.search(r"^\s*([ABCD])", txt, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "?" # If the model generated irrelevant content, mark it as '?'

def load_model_and_tokenizer(base_model_path, adapter_path, local_only=True):
    """(V2) Load the model using 4-bit quantization."""

    if adapter_path and not PEFT_AVAILABLE:
        raise ImportError("Detected --adapter, but the 'peft' library is not installed. Please run: pip install peft")

    print(f"Loading base tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=True,
        local_files_only=local_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # IMPORTANT: model.generate() must use "left" padding.
    tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model from: {base_model_path} (with 4-bit quantization)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=local_only
    )

    if adapter_path:
        print(f"Loading and applying LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=local_only)
        print("Adapter applied.")

    model.eval()
    return model, tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="Base model path (e.g., llama2)")
    ap.add_argument("--adapter", default="", help="LoRA adapter directory (e.g., out_llama2_official_lora)")
    ap.add_argument("--infile", required=True, help="data/official_infer/..._prompts.jsonl")
    ap.add_argument("--outfile", required=True, help="Output .jsonl file for predictions")
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size for generation (can be increased, e.g., 4 or 8)")
    ap.add_argument("--local_files_only", action="store_true", help="Force using local files only")
    args = ap.parse_args()

    args.local_files_only = True # Force local-only

    if not args.adapter:
        print("WARN: No --adapter provided; evaluating with --base_model (base) directly.")

    model, tok = load_model_and_tokenizer(
        args.base_model,
        args.adapter,
        args.local_files_only
    )
    device = model.device

    # Load prompts
    prompts = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            prompts.append((o["id"], o["prompt"])) # (id, prompt_text)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    w = open(args.outfile, "w", encoding="utf-8")

    total_processed = 0
    # This is a standard generation loop
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i+args.batch_size]
        batch_ids = [x[0] for x in batch]
        batch_texts = [x[1] for x in batch]

        # Tokenize
        inputs = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512 # Truncate context
        ).to(device)

        # Run generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,  # We only need one token (A/B/C/D)
                do_sample=False,   # Greedy decoding (fastest)
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )

        # Decode only the *newly generated* tokens
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        decoded_texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

        # Write results
        for ex_id, gen_text in zip(batch_ids, decoded_texts):
            letter = extract_letter(gen_text) # Extract the letter
            rec = {"id": ex_id, "pred_letter": letter, "gen_text": gen_text.strip()}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total_processed += len(batch)
        print(f"Processed {total_processed} / {len(prompts)}")

    w.close()
    print(f"Saved: {args.outfile}  total={len(prompts)}")

if __name__ == "__main__":
    main()
