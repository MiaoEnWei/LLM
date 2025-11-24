# scripts/train/train_prompt_tuning_llama2.py
# ------------------------------------------------------------------
# (V1) Llama-2 提示词调优 (Prompt Tuning) 训练脚本
# - 使用 4-bit QLoRA 加载以节省显存
# ------------------------------------------------------------------

# --- 1. 导入 (Import) 库 ---
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
    BitsAndBytesConfig # <--- ！！！(Llama-2) 关键：导入 4-bit 配置！！！
)
from peft import (
    PromptTuningConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training # <--- ！！！(Llama-2) 关键：导入 K-bit 准备工具！！！
)

LETTERS = "ABCD"

# --- 2. 设置随机种子的辅助函数 ---
def set_seed(seed: int):
    """一个辅助函数，用来固定所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 3. 定义命令行参数 ---
def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True, help="基座模型路径 (e.g., /media/miaoen/ad.../LLM/llama2)")
    ap.add_argument("--train_file", type=str, required=True, help="训练用的 .jsonl 文件 (e.g., data/official_instruct/medmcqa_train.jsonl)")
    ap.add_argument("--output_dir", type=str, required=True, help="模型检查点和适配器的保存目录 (e.g., out_llama2_official_prompt_tuning)")
    
    # --- 训练参数 ---
    ap.add_argument("--epochs", type=int, default=1, help="训练轮数")
    ap.add_argument("--batch_size", type=int, default=1, help="Llama-2 7B 在 12G 显存上必须用 1")
    ap.add_argument("--grad_accum", type=int, default=16, help="梯度累积步数 (有效批大小 = 1 * 16 = 16)")
    ap.add_argument("--lr", type=float, default=2e-4, help="学习率 (Llama-2 QLoRA 通常用 2e-4)") # <--- (Llama-2) 修改了学习率
    
    # --- Seed 参数 ---
    ap.add_argument("--seed", type=int, default=42, help="设置一个固定的随机种子 (用于可复现性)")
    
    # --- Prompt Tuning (提示词调优) 特定参数 ---
    ap.add_argument("--num_virtual_tokens", type=int, default=10, help="要学习的“虚拟提示词”的长度 (token 数量)")
    
    return ap.parse_args()

# --- 4. 主函数 (Main) ---
def main():
    if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # 4a. 解析命令行参数
    args = build_args()
    
    # 4b. 调用 set_seed 函数
    set_seed(args.seed)

    # --- 4c. 加载 Tokenizer (分词器) ---
    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        use_fast=True, 
        local_files_only=True # 只用本地
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left' # SFT 推荐左填充
    print("Tokenizer loaded.")

    # --- 4d. 加载数据集 ---
    print(f"Loading dataset from {args.train_file}")
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    print("Tokenizing dataset (this may take a moment)...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"Dataset loaded and tokenized, {len(tokenized_dataset)} training examples.")

    # --- 4e. 加载模型 (Llama-2) ---
    print(f"Loading model from {args.model_name_or_path} (with 4-bit QLoRA)")
    
    # ！！！(Llama-2) 关键：配置 4-bit QLoRA ！！！
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # 使用 NF4 量化
        bnb_4bit_compute_dtype=torch.bfloat16, # 计算时使用 bfloat16
        bnb_4bit_use_double_quant=True, # 启用双重量化
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config, # <--- 传入 4-bit 配置
        device_map="auto", # 自动处理 GPU
        local_files_only=True # 确保只用本地的 Llama-2
    )
    model.config.use_cache = False
    
    # ！！！(Llama-2) 关键：为 K-bit 训练做准备！！！
    print("Preparing model for 4-bit training...")
    model = prepare_model_for_kbit_training(model)

    # --- 4f. ！！！关键：配置 Prompt Tuning (提示词调优)！！！ ---
    print("Setting up Prompt Tuning config...")
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text="Answer the following medical question with only one letter (A, B, C, or D):",
        tokenizer_name_or_path=args.model_name_or_path
    )
    
    # 4g. 应用配置 (自动冻结 Llama-2)
    model = get_peft_model(model, peft_config)
    
    print("Prompt Tuning adapter applied to model.")
    model.print_trainable_parameters() # (可训练参数会非常非常少)

    # --- 4h. 配置训练参数 (TrainingArguments) ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=100,
        save_strategy="epoch",
        fp16=False, # QLoRA 推荐使用 bf16
        bf16=True,  # ！！！(Llama-2) 启用 bf16
        optim="paged_adamw_8bit", # <--- ！！！(Llama-2) QLoRA 必须用 paged optimizers
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False, # 修复 Windows Bug
    )

    # --- 4i. 初始化标准 Trainer (训练器) ---
    print("Initializing Trainer...")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, # Causal LM 模式
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # --- 4j. 开始训练 ---
    print("Starting training...")
    trainer.train()

    # --- 4k. 保存最终适配器 ---
    print(f"Training complete. Saving final Prompt Tuning adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)

# --- 5. 脚本入口 ---
if __name__ == "__main__":
    main()