# scripts/train/train_prompt_tuning.py
# ------------------------------------------------------------------
# (V2) Prompt Tuning training script for GPT-2
# - Fixes the Windows crash bug caused by 'no accelerator found'
# - Lowers the default learning rate to 0.03 for better stability
# ------------------------------------------------------------------

import argparse
import os
import random
import numpy as np
import torch

from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    PromptTuningConfig,
    get_peft_model,
    TaskType,
)

LETTERS = "ABCD"


def set_seed(seed: int):
    """Fix all random seeds to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Base model path (e.g., ./gpt2)",
    )
    ap.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training .jsonl file (e.g., data/official_instruct/medmcqa_train.jsonl)",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and adapter (e.g., out_gpt2_official_prompt_tuning)",
    )

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size")
    ap.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    ap.add_argument(
        "--lr",
        type=float,
        default=0.03,
        help="Learning rate (Prompt Tuning is often 0.01~0.3; defaulting to 0.03 is more stable)",
    )

    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # Prompt Tuning specific parameters
    ap.add_argument(
        "--num_virtual_tokens",
        type=int,
        default=10,
        help="Length of the learned 'virtual prompt' (number of tokens)",
    )

    return ap.parse_args()


def main():
    # Avoid CUDA memory fragmentation
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    args = build_args()
    set_seed(args.seed)

    # ---- Tokenizer ----
    print(f"[PromptTuning] Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[PromptTuning] tokenizer.pad_token set to eos_token ({tokenizer.eos_token})")

    print("[PromptTuning] Tokenizer loaded.")

    # ---- Dataset ----
    print(f"[PromptTuning] Loading dataset from {args.train_file}")
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    # Now dataset columns are: ['id', 'input', 'output', 'final_decision']

    # First, merge input + output into a single "text" field
    def merge_to_text(examples):
        texts = []
        inputs = examples.get("input", [])
        outputs = examples.get("output", [])
        for inp, out in zip(inputs, outputs):
            # Concatenate into a full text: "instruction + reference answer"
            text = (str(inp).rstrip() + "\n" + str(out).lstrip()).strip()
            texts.append(text)
        return {"text": texts}

    print("[PromptTuning] Building 'text' field from input+output...")
    dataset = dataset.map(
        merge_to_text,
        batched=True,
        remove_columns=dataset.column_names,  # Drop id/input/output/final_decision; keep only text
    )
    # Now dataset has a single column: ['text']

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding=False,  # Hand off padding to DataCollator
        )

    print("[PromptTuning] Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],  # Keep only tokenized fields
    )
    print(f"[PromptTuning] Dataset loaded and tokenized, {len(tokenized_dataset)} training examples.")

    # ★★★ Key: subsample to avoid an excessive number of steps ★★★
    MAX_TRAIN_SAMPLES = 40000  # You can try 20000 first, then increase if needed
    if len(tokenized_dataset) > MAX_TRAIN_SAMPLES:
        print(f"[PromptTuning] Subsampling to first {MAX_TRAIN_SAMPLES} examples for efficiency.")
        tokenized_dataset = tokenized_dataset.select(range(MAX_TRAIN_SAMPLES))

    print(
        f"[PromptTuning] Dataset loaded and tokenized, "
        f"{len(tokenized_dataset)} training examples."
    )

    # ---- Model (base GPT-2) ----
    print("[PromptTuning] Loading base model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False
    # Align with tokenizer pad_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # ---- Prompt Tuning config ----
    print("[PromptTuning] Setting up Prompt Tuning config...")
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text=(
            "Answer the following medical question with only one letter (A, B, C, or D):"
        ),
        tokenizer_name_or_path=args.model_name_or_path,
    )

    model = get_peft_model(model, peft_config)
    print("[PromptTuning] Prompt Tuning adapter applied to model.")
    model.print_trainable_parameters()

    print("[PromptTuning] Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ---- TrainingArguments ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=100,
        logging_strategy="steps",
        save_strategy="epoch",
        save_total_limit=3,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Added
    )

    # ---- Trainer ----
    print("[PromptTuning] Initializing Trainer...")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # ---- Train ----
    print("[PromptTuning] Starting training...")
    trainer.train()

    # ---- Save adapter ----
    print(
        f"[PromptTuning] Training complete. Saving final Prompt Tuning adapter to {args.output_dir}"
    )
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
