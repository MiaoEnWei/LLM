# scripts/predict_logits_lora.py
# ---------------------------------------------------------
# V3 - 通用 logits 多选打分脚本
# - 支持 GPT-2 / LLaMA 等 Causal LM
# - 支持 PEFT 适配器 (LoRA / Prompt Tuning)
# - 默认使用 fp16，全模型；可选 --use_4bit 用于大模型
# - 输入: 每行 {"id": ..., "prompt": "...Answer (A, B, C, or D):"}
# - 输出: 每行 {"id": ..., "pred_letter": "A/B/C/D", "score": logprob}
# ---------------------------------------------------------

import argparse
import json
import os
import torch
from torch.nn.functional import log_softmax
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

LETTER = "ABCD"


def load_model_and_tokenizer(base_model_path, adapter_path, local_only=True, use_4bit=False):
    """加载基座模型 + 可选 PEFT 适配器"""
    if adapter_path and not PEFT_AVAILABLE:
        raise ImportError("检测到 --adapter，但未安装 'peft' 库，请先: pip install peft")

    print(f"[predict] Loading tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=True,
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 方便截断 + 对齐

    print(f"[predict] Loading base model from: {base_model_path}")
    if use_4bit:
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError("需要 bitsandbytes 才能使用 --use_4bit，请先安装。")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quant_config,
            device_map="auto",
            local_files_only=local_only,
        )
    else:
        # GPT-2 / 小模型走这个分支即可
        kwargs = {"local_files_only": local_only}
        if torch.cuda.is_available():
            kwargs.update(
                dict(
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **kwargs,
        )

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if adapter_path:
        print(f"[predict] Loading PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=local_only,
        )
        print("[predict] Adapter applied.")

    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_model",
        required=True,
        help="基座模型路径 (e.g., ./gpt2, ./llama2)",
    )
    ap.add_argument(
        "--adapter",
        default="",
        help="PEFT 适配器目录 (e.g., out_gpt2_official_prompt_tuning)",
    )
    ap.add_argument(
        "--infile",
        required=True,
        help="输入 prompts.jsonl (每行包含 id / prompt)",
    )
    ap.add_argument(
        "--outfile",
        required=True,
        help="输出 preds.jsonl 文件路径",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="评测批大小 (大模型可调小)",
    )
    ap.add_argument(
        "--local_files_only",
        action="store_true",
        help="强制只使用本地文件",
    )
    ap.add_argument(
        "--use_4bit",
        action="store_true",
        help="对大模型使用 4-bit 量化 (GPT-2 一般不需要)",
    )

    args = ap.parse_args()

    # 你本地一直用本地路径，这里直接锁死为 True
    args.local_files_only = True

    if not args.adapter:
        print("[WARN] 未提供 --adapter，将只使用基座模型进行评测。")

    model, tok = load_model_and_tokenizer(
        args.base_model,
        args.adapter,
        local_only=args.local_files_only,
        use_4bit=args.use_4bit,
    )
    device = model.device

    # 读取 prompts
    prompts = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            pid = o["id"]
            p = o["prompt"]
            prompts.append((pid, p))

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    w = open(args.outfile, "w", encoding="utf-8")

    # 预先拿到 A/B/C/D 的 token id
    label_tokens = {}
    for L in LETTER:
        ids = tok.encode(L, add_special_tokens=False)
        if len(ids) == 0:
            raise ValueError(f"无法给字母 {L} 编码成 token id")
        label_tokens[L] = ids[0]  # 对 'A','B','C','D' 来说基本都是单 token
    print(f"[predict] Label token ids: {label_tokens}")

    total = len(prompts)
    processed = 0

    for i in range(0, total, args.batch_size):
        batch = prompts[i : i + args.batch_size]
        ids = [x[0] for x in batch]
        texts = [x[1] for x in batch]

        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, T, V]
            logp = log_softmax(logits, dim=-1)

        # 对每个样本，找最后一个非 pad 的位置，对应“下一个 token”的分布
        # 我们用该位置的 logits 来评估 A/B/C/D 的 logprob
        last_indices = attention_mask.sum(dim=1) - 1  # [B]

        for b_idx, ex_id in enumerate(ids):
            pos = last_indices[b_idx].item()
            # 对应位置的 logprob 向量
            lp_vec = logp[b_idx, pos]  # [V]

            best_L = None
            best_score = float("-inf")

            for L in LETTER:
                tid = label_tokens[L]
                s = lp_vec[tid].item()
                if s > best_score:
                    best_score = s
                    best_L = L

            w.write(
                json.dumps(
                    {
                        "id": ex_id,
                        "pred_letter": best_L,
                        "score": best_score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        processed += len(batch)
        if processed % (args.batch_size * 10) == 0 or processed == total:
            print(f"[predict] Processed {processed} / {total}")

    w.close()
    print(f"[predict] Saved preds to: {args.outfile}  total={total}")


if __name__ == "__main__":
    main()
