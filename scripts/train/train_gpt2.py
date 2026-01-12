# scripts/train_lora.py
# This is a more stable LoRA training script that does not depend on the "trl" library

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
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True, help="Base model path (e.g., ./gpt2)")
    ap.add_argument("--train_file", type=str, required=True, help="Training .jsonl file (e.g., data/official_instruct/medmcqa_train.jsonl)")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and adapters (e.g., out_gpt2_official_lora)")
    
    # --- Training hyperparameters ---
    ap.add_argument("--epochs", type=int, default=1, help="Number of epochs (for ~180k samples, 1 epoch is usually enough)")
    ap.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size (use 1 or 2 if VRAM is limited)")
    ap.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (effective batch size = batch_size * grad_accum)")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate (LoRA typically uses 2e-4 or 3e-4)")
    
    # --- LoRA hyperparameters ---
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA r (rank)")
    ap.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    args = ap.parse_args()

    # --- 1. Load tokenizer ---
    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    # --- 2. Load dataset ---
    print(f"Loading dataset from {args.train_file}")
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    
    # Helper: tokenize the 'text' field
    def tokenize_function(examples):
        # We only truncate and do not pad; the DataCollator will handle padding
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    print("Tokenizing dataset (this may take a moment)...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"Dataset loaded and tokenized, {len(tokenized_dataset)} training examples.")

    # --- 3. Load model (use 8-bit to reduce VRAM usage) ---
    print(f"Loading model from {args.model_name_or_path}")
    
    # Tell bitsandbytes to load in 8-bit
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config, # Pass the 8-bit config
        device_map="auto", # Automatically place layers on GPU/CPU as needed
        local_files_only=True # Ensure only local files are used (e.g., ./gpt2)
    )
    model.config.use_cache = False
    
    # --- 4. Configure PEFT (LoRA) ---
    print("Preparing model for 8-bit training...")
    model = prepare_model_for_kbit_training(model)

    print("Setting up LoRA config...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"], # For GPT-2
    )
    
    # Manually apply LoRA to the model
    model = get_peft_model(model, peft_config)
    print("LoRA adapter applied to model.")
    model.print_trainable_parameters()

    # --- 5. Training arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=100,
        save_strategy="epoch", # Save once per epoch
        optim="paged_adamw_8bit",
        fp16=True, # Enable fp16
        report_to="none", # Disable wandb/tensorboard
        seed=42,  # <--- !!! Add a fixed seed here for reproducibility !!!
    )

    # --- 6. Initialize standard Trainer ---
    print("Initializing Trainer...")
    
    # We need a Data Collator to handle batching and padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, # Ensure this is causal LM mode
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # --- 7. Train ---
    print("Starting training...")
    trainer.train()

    # --- 8. Save final adapter ---
    print(f"Training complete. Saving final LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir) # Save only the LoRA adapter

if __name__ == "__main__":
    main()
