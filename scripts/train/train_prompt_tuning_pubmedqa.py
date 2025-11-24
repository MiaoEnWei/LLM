# scripts/train/train_prompt_tuning_pubmedqa.py
# -------------------------------------------------------------
# GPT-2 在 PubMedQA (pqa_labeled) 上的 Prompt Tuning 训练脚本
# - 从 parquet 读取 question / context / long_answer / final_decision
# - 构造指令风格 text 字段，然后做 Prompt Tuning
# -------------------------------------------------------------

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True,
                    help="基座模型路径 (e.g., ./gpt2)")
    ap.add_argument("--parquet", type=str, required=True,
                    help="PubMedQA parquet 文件路径 (e.g., data/pubmedqa_hf/pqa_labeled/train-00000-of-00001.parquet)")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Prompt Tuning adapter 输出目录 (e.g., out_gpt2_pubmedqa_prompt_tuning)")

    # 训练超参
    ap.add_argument("--epochs", type=int, default=5,
                    help="训练轮数 (PubMedQA 样本不多，可以多跑几圈)")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="per-device 训练批大小")
    ap.add_argument("--grad_accum", type=int, default=8,
                    help="梯度累积步数 (有效 batch = batch_size * grad_accum)")
    ap.add_argument("--lr", type=float, default=0.01,
                    help="学习率 (Prompt Tuning 通常 1e-2 左右)")

    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子")
    ap.add_argument("--num_virtual_tokens", type=int, default=20,
                    help="虚拟提示 token 数 (例如 10 / 20 / 50)")

    return ap.parse_args()


def flatten_context(c, max_chars: int = 4000) -> str:
    """把 PubMedQA 的 context 对象拍平成字符串."""
    if c is None:
        return ""
    # pqa_labeled 里 context 一般是 {"contexts": [str, str, ...]}
    if isinstance(c, dict):
        ctxs = c.get("contexts", [])
        if isinstance(ctxs, list):
            out = " ".join(str(x) for x in ctxs)
        else:
            out = str(ctxs)
    elif isinstance(c, list):
        out = " ".join(str(x) for x in c)
    elif isinstance(c, str):
        out = c
    else:
        out = str(c)
    return out[:max_chars]


def main():
    # 避免显存碎片
    if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    args = build_args()
    set_seed(args.seed)

    # 1) tokenizer
    print(f"[train_pubmedqa] Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("[train_pubmedqa] Tokenizer loaded.")

    # 2) 加载 PubMedQA parquet -> datasets
    print(f"[train_pubmedqa] Loading PubMedQA parquet from {args.parquet}")
    dataset = load_dataset(
        "parquet",
        data_files={"train": args.parquet},
    )["train"]

    # 预期字段: question, context, long_answer, final_decision
    def build_text(example):
        q = (example.get("question") or "").strip()
        ctx_obj = example.get("context")
        ctx = flatten_context(ctx_obj, max_chars=4000)

        long_ans = (example.get("long_answer") or "").strip()
        dec = (example.get("final_decision") or "").strip().lower()

        # 规范决策词
        if dec.startswith("y"):
            dec = "Yes"
        elif dec.startswith("n"):
            dec = "No"
        else:
            dec = "Maybe"

        # 和 eval_pubmedqa_gen.py 一致的指令风格
        prompt_lines = [
            "You are a biomedical QA assistant.",
            f"Question: {q}",
        ]
        if ctx:
            prompt_lines.append(f"Context: {ctx}")
        prompt_lines.append(
            "Provide a concise rationale (1-2 sentences) and end with a final decision word among [Yes, No, Maybe]."
        )
        prompt = "\n".join(prompt_lines) + "\n\n"

        # 目标：参考结论 + 决策词
        target = f"{long_ans} Decision: {dec}"

        text = prompt + target
        return {"text": text}

    print("[train_pubmedqa] Building instruction-style text field...")
    dataset_text = dataset.map(
        build_text,
        remove_columns=dataset.column_names,
        desc="Adding text field",
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,  # 让 DataCollator 处理 padding
        )

    print("[train_pubmedqa] Tokenizing dataset...")
    tokenized_dataset = dataset_text.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    print(f"[train_pubmedqa] Tokenized samples: {len(tokenized_dataset)}")

    # 3) 加载 GPT-2
    print(f"[train_pubmedqa] Loading base model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False

    # 4) Prompt Tuning 配置
    print("[train_pubmedqa] Setting up Prompt Tuning config...")
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text=(
            "You are a biomedical QA assistant. "
            "Provide a concise rationale (1-2 sentences) and end with a final decision word among [Yes, No, Maybe]."
        ),
        tokenizer_name_or_path=args.model_name_or_path,
    )

    model = get_peft_model(model, peft_config)
    print("[train_pubmedqa] Prompt Tuning adapter applied.")
    model.print_trainable_parameters()

    # 5) TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,
    )

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

    print("[train_pubmedqa] Starting training...")
    trainer.train()

    print(f"[train_pubmedqa] Training complete. Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
