# scripts/train/train_prompt_tuning.py
# ------------------------------------------------------------------
# (V2) GPT-2 提示词调优 (Prompt Tuning) 训练脚本
# - 修复了 'no accelerator found' 导致的 Windows 崩溃 Bug
# - 默认学习率调低为 0.03，更稳定
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
    """固定所有随机种子，保证可复现性"""
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
        help="基座模型路径 (e.g., ./gpt2)",
    )
    ap.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="训练用的 .jsonl 文件 (e.g., data/official_instruct/medmcqa_train.jsonl)",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="模型检查点和适配器的保存目录 (e.g., out_gpt2_official_prompt_tuning)",
    )

    # 训练超参
    ap.add_argument("--epochs", type=int, default=3, help="训练轮数")
    ap.add_argument("--batch_size", type=int, default=4, help="Per-device 训练批大小")
    ap.add_argument("--grad_accum", type=int, default=8, help="梯度累积步数")
    ap.add_argument(
        "--lr",
        type=float,
        default=0.03,
        help="学习率 (Prompt Tuning 通常 0.01~0.3，这里默认 0.03 更稳)",
    )

    ap.add_argument("--seed", type=int, default=42, help="随机种子")

    # Prompt Tuning 特定参数
    ap.add_argument(
        "--num_virtual_tokens",
        type=int,
        default=10,
        help="要学习的“虚拟提示词”的长度 (token 数量)",
    )

    return ap.parse_args()


def main():
    # 避免显存碎片
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
    # 现在 dataset 的列是: ['id', 'input', 'output', 'final_decision']

    # 先把 input + output 合成一个 text 字段
    def merge_to_text(examples):
        texts = []
        inputs = examples.get("input", [])
        outputs = examples.get("output", [])
        for inp, out in zip(inputs, outputs):
            # 拼接成「指令 + 参考答案」的完整文本
            text = (str(inp).rstrip() + "\n" + str(out).lstrip()).strip()
            texts.append(text)
        return {"text": texts}

    print("[PromptTuning] Building 'text' field from input+output...")
    dataset = dataset.map(
        merge_to_text,
        batched=True,
        remove_columns=dataset.column_names,  # 删除原来的 id/input/output/final_decision，只保留 text
    )
    # 现在 dataset 只有一列: ['text']

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding=False,  # 交给 DataCollator
        )

    print("[PromptTuning] Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],  # 只保留 tokenized 字段
    )
    print(f"[PromptTuning] Dataset loaded and tokenized, {len(tokenized_dataset)} training examples.")

    # ★★★ 关键：下采样，避免 20 万步这么夸张 ★★★
    MAX_TRAIN_SAMPLES = 40000  # 你可以先试 20000，看时间，再决定要不要加
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
    # 配合 tokenizer 的 pad_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # ---- Prompt Tuning 配置 ----
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
    gradient_checkpointing=True,   # 新增这一行
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
