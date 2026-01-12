# scripts/train_lora_llama2.py
# This is a QLoRA (4-bit) training script optimized specifically for Llama-2-7b

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig  # We need this to configure 4-bit
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def main():
    ap = argparse.ArgumentParser()
    # Note: model_name_or_path should point to the Llama-2-7b path
    ap.add_argument("--model_name_or_path", type=str, required=True, help="Base model path (e.g., ./Llama-2-7b-hf)")
    ap.add_argument("--train_file", type=str, required=True, help="Training .jsonl file (e.g., data/official_instruct/medmcqa_train.jsonl)")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and adapters (e.g., out_llama2_official_lora)")

    # --- Training arguments ---
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1, help="For Llama-2 7B on 12GB VRAM, you must use 1")
    ap.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation (effective batch size = 1 * 16 = 16)")
    ap.add_argument("--lr", type=float, default=2e-4)

    # --- LoRA arguments ---
    ap.add_argument("--lora_r", type=int, default=64, help="For Llama-2, r=64 is recommended")
    ap.add_argument("--lora_alpha", type=int, default=16, help="For Llama-2, alpha=16 is recommended")
    ap.add_argument("--lora_dropout", type=float, default=0.1)

    args = ap.parse_args()

    # --- 1. Load tokenizer ---
    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        # Llama-2 has no pad token, so we must set it manually
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # Left padding is recommended for SFT
    print("Tokenizer loaded.")

    # --- 2. Load dataset ---
    print(f"Loading dataset from {args.train_file}")
    dataset = load_dataset("json", data_files=args.train_file, split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"Dataset loaded and tokenized, {len(tokenized_dataset)} training examples.")

    # --- 3. Load model (4-bit QLoRA) ---
    print(f"Loading model from {args.model_name_or_path}")

    # !!! Key: configure 4-bit QLoRA !!!
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Use NF4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computation
        bnb_4bit_use_double_quant=True,  # Enable double quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,  # Pass the 4-bit config
        device_map="auto",  # Automatically handle GPU placement
        local_files_only=True  # Ensure we only use the local Llama-2 files
    )
    model.config.use_cache = False

    # --- 4. Configure PEFT (LoRA) ---
    print("Preparing model for 4-bit training...")
    model = prepare_model_for_kbit_training(model)

    print("Setting up LoRA config...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Llama-2 target modules differ from GPT-2
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)
    print("LoRA adapter applied to model.")
    model.print_trainable_parameters()

    # --- 5. Configure training arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=100,
        save_strategy="epoch",
        optim="paged_adamw_8bit",  # QLoRA requires paged optimizers
        fp16=False,  # QLoRA recommends bf16
        bf16=True,   # !!! Enable bf16 (if your RTX 3060 supports it)
        report_to="none",
    )

    # --- 6. Initialize standard Trainer ---
    print("Initializing Trainer...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # --- 7. Start training ---
    print("Starting training...")
    trainer.train()

    # --- 8. Save final adapter ---
    print(f"Training complete. Saving final LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
