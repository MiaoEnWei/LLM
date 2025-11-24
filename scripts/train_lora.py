# scripts/train_lora.py
# 这是一个不依赖 "trl" 库的、更稳定的 LoRA 训练脚本

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
    ap.add_argument("--model_name_or_path", type=str, required=True, help="基座模型路径 (e.g., ./gpt2)")
    ap.add_argument("--train_file", type=str, required=True, help="训练用的 .jsonl 文件 (e.g., data/official_instruct/medmcqa_train.jsonl)")
    ap.add_argument("--output_dir", type=str, required=True, help="模型检查点和适配器的保存目录 (e.g., out_gpt2_official_lora)")
    
    # --- 训练参数 ---
    ap.add_argument("--epochs", type=int, default=1, help="训练轮数 (对于18万数据，1轮就够了)")
    ap.add_argument("--batch_size", type=int, default=2, help="Per-device 训练批大小 (如果显存小就用 1 或 2)")
    ap.add_argument("--grad_accum", type=int, default=8, help="梯度累积步数 (有效批大小 = batch_size * grad_accum)")
    ap.add_argument("--lr", type=float, default=2e-4, help="学习率 (LoRA 通常用 2e-4 或 3e-4)")
    
    # --- LoRA 参数 ---
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA r (rank)")
    ap.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    args = ap.parse_args()

    # --- 1. 加载 Tokenizer ---
    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    # --- 2. 加载数据集 ---
    print(f"Loading dataset from {args.train_file}")
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    
    # 辅助函数：将 'text' 字段标记化
    def tokenize_function(examples):
        # 我们只截断，不填充(padding)，DataCollator 会处理填充
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    print("Tokenizing dataset (this may take a moment)...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"Dataset loaded and tokenized, {len(tokenized_dataset)} training examples.")

    # --- 3. 加载模型 (使用 8-bit 优化节省显存) ---
    print(f"Loading model from {args.model_name_or_path}")
    
    # 告诉 bitsandbytes 用 8-bit 加载
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config, # 传入 8-bit 配置
        device_map="auto", # 自动处理 GPU
        local_files_only=True # 确保只用本地的 ./gpt2
    )
    model.config.use_cache = False
    
    # --- 4. 配置 PEFT (LoRA) ---
    print("Preparing model for 8-bit training...")
    model = prepare_model_for_kbit_training(model)

    print("Setting up LoRA config...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"], # 适配 GPT-2
    )
    
    # 手动应用 LoRA 到模型
    model = get_peft_model(model, peft_config)
    print("LoRA adapter applied to model.")
    model.print_trainable_parameters()

    # --- 5. 配置训练参数 ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=100,
        save_strategy="epoch", # 每个 epoch 保存一次
        optim="paged_adamw_8bit",
        fp16=True, # 启用 fp16
        report_to="none", # 关闭 wandb/tensorboard
        seed=42,  # <--- ！！！在这里添加一个固定的种子！！！
    )

    # --- 6. 初始化标准 Trainer ---
    print("Initializing Trainer...")
    
    # 我们需要一个数据整理器（Data Collator）来处理批次和填充
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, # 确保这是 Causal LM 模式
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # --- 7. 开始训练 ---
    print("Starting training...")
    trainer.train()

    # --- 8. 保存最终适配器 ---
    print(f"Training complete. Saving final LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir) # 只保存 LoRA 适配器

if __name__ == "__main__":
    main()