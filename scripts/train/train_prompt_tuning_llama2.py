# scripts/train/train_prompt_tuning_llama2.py
# ------------------------------------------------------------------
# (V1) Prompt Tuning training script for Llama-2
# - Loads with 4-bit QLoRA to save VRAM
# ------------------------------------------------------------------

# --- 1. Import libraries ---
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
    BitsAndBytesConfig  # <--- (Llama-2) Key: import 4-bit configuration
)
from peft import (
    PromptTuningConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training  # <--- (Llama-2) Key: import k-bit preparation utility
)

LETTERS = "ABCD"

# --- 2. Helper function to set random seeds ---
def set_seed(seed: int):
    """A helper function to fix all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 3. Define CLI arguments ---
def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Base model path (e.g., /media/miaoen/ad.../LLM/llama2)",
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
        help="Directory to save checkpoints and adapter (e.g., out_llama2_official_prompt_tuning)",
    )

    # --- Training parameters ---
    ap.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=1, help="For Llama-2 7B on 12GB VRAM, batch_size must be 1")
    ap.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps (effective batch = 1 * 16 = 16)")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate (for Llama-2 QLoRA, commonly ~2e-4)")  # (Llama-2) modified LR

    # --- Seed parameter ---
    ap.add_argument("--seed", type=int, default=42, help="Set a fixed random seed (for reproducibility)")

    # --- Prompt Tuning specific parameter ---
    ap.add_argument("--num_virtual_tokens", type=int, default=10, help="Length of the learned 'virtual prompt' (number of tokens)")

    return ap.parse_args()

# --- 4. Main ---
def main():
    if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # 4a. Parse CLI arguments
    args = build_args()

    # 4b. Set seeds
    set_seed(args.seed)

    # --- 4c. Load tokenizer ---
    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        local_files_only=True  # local files only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # left padding is recommended for SFT
    print("Tokenizer loaded.")

    # --- 4d. Load dataset ---
    print(f"Loading dataset from {args.train_file}")
    dataset = load_dataset("json", data_files=args.train_file, split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    print("Tokenizing dataset (this may take a moment)...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"Dataset loaded and tokenized, {len(tokenized_dataset)} training examples.")

    # --- 4e. Load model (Llama-2) ---
    print(f"Loading model from {args.model_name_or_path} (with 4-bit QLoRA)")

    # (Llama-2) Key: configure 4-bit QLoRA
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # use NF4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # use bfloat16 for computation
        bnb_4bit_use_double_quant=True,     # enable double quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,  # pass 4-bit config
        device_map="auto",                        # automatically place layers on GPU/CPU
        local_files_only=True                     # ensure local-only Llama-2
    )
    model.config.use_cache = False

    # (Llama-2) Key: prepare for k-bit training
    print("Preparing model for 4-bit training...")
    model = prepare_model_for_kbit_training(model)

    # --- 4f. Key: configure Prompt Tuning ---
    print("Setting up Prompt Tuning config...")
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text="Answer the following medical question with only one letter (A, B, C, or D):",
        tokenizer_name_or_path=args.model_name_or_path
    )

    # 4g. Apply PEFT config (Llama-2 is automatically frozen)
    model = get_peft_model(model, peft_config)

    print("Prompt Tuning adapter applied to model.")
    model.print_trainable_parameters()  # trainable params will be extremely small

    # --- 4h. TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=100,
        save_strategy="epoch",
        fp16=False,                 # QLoRA commonly uses bf16
        bf16=True,                  # (Llama-2) enable bf16
        optim="paged_adamw_8bit",   # (Llama-2) QLoRA requires paged optimizers
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,  # mitigates a Windows issue
    )

    # --- 4i. Initialize Trainer ---
    print("Initializing Trainer...")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM mode
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # --- 4j. Start training ---
    print("Starting training...")
    trainer.train()

    # --- 4k. Save final adapter ---
    print(f"Training complete. Saving final Prompt Tuning adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)

# --- 5. Script entrypoint ---
if __name__ == "__main__":
    main()
