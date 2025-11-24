# scripts/train/train_prompt_tuning_llama2_pubmedqa.py
# ---------------------------------------------------------
# LLaMA-2 在 PubMedQA (pqa_labeled) 上的 Prompt Tuning 训练脚本
# - 使用 4-bit QLoRA + PromptTuning (虚拟 token)
# - 从本地 parquet 读取 pqa_labeled/train-00000-of-00001.parquet
# ---------------------------------------------------------

import argparse
import os
import random
import numpy as np
import torch
import pyarrow.parquet as pq

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    PromptTuningConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# ---------------------------------------------------------
# 0. Seed
# ---------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# 1. CLI 参数
# ---------------------------------------------------------

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="LLaMA-2 基座模型路径，例如 ./llama2",
    )
    ap.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="PubMedQA pqa_labeled parquet 文件路径",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Prompt Tuning 适配器保存目录，例如 out_llama2_pubmedqa_prompt_tuning",
    )

    # 训练超参
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="LLaMA2-7B + 4bit 通常只能 batch=1",
    )
    ap.add_argument(
        "--grad_accum",
        type=int,
        default=16,
        help="梯度累积步数（有效 batch = batch_size * grad_accum）",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="学习率，QLoRA 常用 2e-4 左右",
    )
    ap.add_argument(
        "--num_virtual_tokens",
        type=int,
        default=20,
        help="Prompt Tuning 的虚拟 token 数量",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=5000,
        help="最多使用多少条样本；0 表示用全部",
    )
    ap.add_argument(
        "--max_ctx_chars",
        type=int,
        default=4000,
        help="拼接 context 时最多截取多少字符",
    )
    return ap.parse_args()


# ---------------------------------------------------------
# 2. PubMedQA 数据处理
# ---------------------------------------------------------

def join_context(c, max_ctx_chars: int) -> str:
    """
    PubMedQA pqa_labeled 的 context 是个 dict，大概长这样：
    {"contexts": ["sentence1", "sentence2", ...]}
    这里把它拍平成一个长字符串，并截断。
    """
    try:
        ctxs = (c or {}).get("contexts", [])
        if not ctxs:
            return ""
        out = " ".join(ctxs)
        return out[:max_ctx_chars]
    except Exception:
        return ""


def build_instruction_text(example, max_ctx_chars: int):
    """
    把 question + context + long_answer 变成一条 instruction-style 文本：
    [系统提示] + Question + Context + Answer
    """
    q = (example.get("question") or "").strip()
    ctx_raw = example.get("context")
    ctx = join_context(ctx_raw, max_ctx_chars=max_ctx_chars)
    ans = (example.get("long_answer") or "").strip()

    text = (
        "You are a biomedical QA assistant.\n"
        f"Question: {q}\n"
        f"Context: {ctx}\n\n"
        "Provide a concise rationale (1-3 sentences) answering the question.\n"
        f"Answer: {ans}"
    )

    return {"text": text}


# ---------------------------------------------------------
# 3. 主函数
# ---------------------------------------------------------

def main():
    # 兼容 Windows / Linux 显存碎片问题
    if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    args = build_args()
    set_seed(args.seed)

    # 3.1 加载 tokenizer
    print(f"[train_pubmedqa_llama] Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 对 causal LM + 左填充更友好
    tokenizer.padding_side = "left"
    print("[train_pubmedqa_llama] Tokenizer loaded.")

    # 3.2 读取 parquet -> pandas -> HF Dataset
    print(f"[train_pubmedqa_llama] Loading parquet from {args.parquet}")
    table = pq.read_table(args.parquet)
    df = table.to_pandas()

    # 只保留必要列，并去掉 NA
    cols = ["question", "context", "long_answer"]
    df = df[cols].dropna().reset_index(drop=True)

    # 按参数控制训练样本数
    if args.max_train_samples and args.max_train_samples > 0:
        n = min(args.max_train_samples, len(df))
        df = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
        print(f"[train_pubmedqa_llama] Using {len(df)} samples for training (shuffled).")
    else:
        print(f"[train_pubmedqa_llama] Using ALL {len(df)} samples for training.")

    dataset = Dataset.from_pandas(df, preserve_index=False)

    print("[train_pubmedqa_llama] Building instruction-style text field...")
    dataset_text = dataset.map(
        lambda ex: build_instruction_text(ex, max_ctx_chars=args.max_ctx_chars),
        desc="Adding text field",
    )

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    print("[train_pubmedqa_llama] Tokenizing dataset...")
    tokenized_dataset = dataset_text.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset_text.column_names,
        desc="Tokenizing",
    )
    print(f"[train_pubmedqa_llama] Tokenized examples: {len(tokenized_dataset)}")

    # 3.3 加载 LLaMA-2 模型（4-bit QLoRA）
    print(f"[train_pubmedqa_llama] Loading model from {args.model_name_or_path} (4-bit QLoRA)")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    print("[train_pubmedqa_llama] Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # 3.4 Prompt Tuning 配置
    print("[train_pubmedqa_llama] Setting up Prompt Tuning config...")
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text=(
            "You are a biomedical QA assistant. "
            "Answer the question based on the given context."
        ),
        tokenizer_name_or_path=args.model_name_or_path,
    )

    model = get_peft_model(model, peft_config)
    print("[train_pubmedqa_llama] Prompt Tuning adapter applied.")
    model.print_trainable_parameters()

    # 3.5 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="epoch",
        fp16=False,
        bf16=True,  # Ampere 以上用 bf16
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print("[train_pubmedqa_llama] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("[train_pubmedqa_llama] Starting training...")
    trainer.train()

    print(f"[train_pubmedqa_llama] Training complete. Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
