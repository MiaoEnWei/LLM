"""
PubMedQA 决策一致性评测（Decision-only ACC）

- 从官方 parquet 读取 question / context / final_decision
- 用 Causal LM（GPT-2 / LLaMA 等）生成【只输出 Yes/No/Maybe】
- 计算 Yes/No/Maybe 的 ACC + 混淆矩阵
"""

import os
import re
import json
import random
import argparse
from collections import Counter

import numpy as np
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging

# 可选：PEFT 适配器支持（Prompt Tuning / LoRA）
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def set_seed(seed: int):
    """尽量做到可复现（CPU/GPU）。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def join_context(c, max_chars: int) -> str:
    """把 context 结构拍平为字符串，只取 contexts 字段并裁剪。"""
    try:
        ctxs = (c or {}).get("contexts", [])
        if not ctxs:
            return ""
        out = " ".join(ctxs)
        return out[:max_chars]
    except Exception:
        return ""


def build_prompt(tok, use_chat_template: bool, q: str, ctx: str) -> str:
    """
    决策-only 的 prompt：要求模型只输出 Yes/No/Maybe
    """
    if use_chat_template and hasattr(tok, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": "You are a biomedical QA assistant."},
            {
                "role": "user",
                "content": (
                    f"Question: {q}\n"
                    f"Context: {ctx}\n\n"
                    "Answer with only one word among [Yes, No, Maybe].\n"
                    "Answer:"
                ),
            },
        ]
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    # 非 chat 模型：直接用 plain prompt
    return (
        "You are a biomedical QA assistant.\n"
        f"Question: {q}\n"
        f"Context: {ctx}\n\n"
        "Answer with only one word among [Yes, No, Maybe].\n"
        "Answer:"
    )


def extract_decision(text: str):
    """
    从生成文本中抽取 yes/no/maybe（大小写不敏感）：
    1) 先从开头附近找（更符合“只输出一个词”的预期）
    2) 再 fallback 到全文最后一次出现
    """
    t = (text or "").strip().lower()

    # 去掉前后引号 / 标点
    t = t.strip(' "\'\n\t.')

    # 试着匹配“开头就是一个词”
    m = re.match(r"^(yes|no|maybe)\b\.?$", t)
    if m:
        return m.group(1)

    # 全文中找所有 yes/no/maybe，取最后一个
    cand = re.findall(r"\b(yes|no|maybe)\b", t)
    return cand[-1] if cand else None


def normalize_decision(x: str) -> str:
    """把 gold decision 统一成 yes/no/maybe 小写三类."""
    s = (str(x) or "").strip().lower()
    if s.startswith("y"):
        return "yes"
    if s.startswith("n"):
        return "no"
    return "maybe"


def decision_metrics(pred_list, gold_list):
    """给出基础 ACC 和混淆统计。"""
    assert len(pred_list) == len(gold_list)
    valid_idx = [i for i, p in enumerate(pred_list) if p is not None]
    if not valid_idx:
        return {"acc": 0.0, "support": 0, "coverage": 0.0, "confusion": {}}

    preds = [pred_list[i] for i in valid_idx]
    golds = [gold_list[i] for i in valid_idx]

    correct = sum(int(p == g) for p, g in zip(preds, golds))
    acc = correct / len(valid_idx)
    coverage = len(valid_idx) / len(pred_list)

    labels = ["yes", "no", "maybe"]
    conf = Counter((g, p) for g, p in zip(golds, preds))
    confusion = {f"{g}->{p}": conf[(g, p)] for g in labels for p in labels}

    return {
        "acc": acc,
        "support": len(valid_idx),
        "coverage": coverage,
        "confusion": confusion,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="pqa_labeled_test.parquet 路径")
    ap.add_argument("--model", required=True, help="HF 模型名或本地权重目录")
    ap.add_argument("--adapter", default="", help="PEFT 适配器目录，可选（Prompt Tuning / LoRA）")
    ap.add_argument("--use_chat_template", action="store_true", help="对 -chat 模型启用 chat 模板")
    ap.add_argument("--limit", type=int, default=500, help="评测样本数（从头部截取）")
    ap.add_argument("--max_new_tokens", type=int, default=4, help="最多生成几个 token（决策任务不需要太多）")
    ap.add_argument("--max_ctx_chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true", help="静默 Transformers 日志")
    ap.add_argument("--local_files_only", action="store_true", help="仅使用本地模型文件")
    args = ap.parse_args()

    if args.quiet:
        hf_logging.set_verbosity_error()

    set_seed(args.seed)

    # 读取 & 对齐数据
    tbl = pq.read_table(args.parquet)
    df_full = tbl.to_pandas()

    needed_cols = ["question", "context", "final_decision"]
    for col in needed_cols:
        if col not in df_full.columns:
            raise ValueError(f"parquet 缺少列: {col}，当前列: {list(df_full.columns)}")

    df = df_full[needed_cols].dropna().head(args.limit)
    df["ctx"] = df["context"].map(lambda c: join_context(c, args.max_ctx_chars))

    questions = df["question"].tolist()
    ctx_list = df["ctx"].tolist()
    gold_decisions = [normalize_decision(x) for x in df["final_decision"].tolist()]

    # 加载模型 & tokenizer
    use_fp16 = torch.cuda.is_available()
    print(f"[DecisionEval] Loading tokenizer from: {args.model}")
    tok = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[DecisionEval] Loading base model from: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=(torch.float16 if use_fp16 else torch.float32),
        device_map="auto" if use_fp16 else None,
        local_files_only=args.local_files_only,
    )
    model.config.pad_token_id = tok.pad_token_id

    # 可选加载 PEFT adapter
    if args.adapter:
        if not PEFT_AVAILABLE:
            raise ImportError("指定了 --adapter，但当前环境未安装 peft，请先 pip install peft")
        print(f"[DecisionEval] Loading PEFT adapter from: {args.adapter}")
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            local_files_only=args.local_files_only,
        )
    model.eval()

    preds = []

    for i, (q, ctx) in enumerate(zip(questions, ctx_list)):
        prompt = build_prompt(tok, args.use_chat_template, q, ctx)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # 决策任务先用贪心
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen = tok.decode(gen_ids, skip_special_tokens=True).strip()
        preds.append(gen)

        if (i + 1) % 20 == 0:
            print(f"[DecisionEval] {i+1}/{len(df)}")

    # 抽取决策 & 计算指标
    pred_decisions = [extract_decision(t) for t in preds]
    metrics = decision_metrics(pred_decisions, gold_decisions)

    print("\n=== PubMedQA Decision-only Evaluation (Yes/No/Maybe) ===")
    print(json.dumps(
        {
            "acc": round(metrics["acc"], 6),
            "support_extracted": metrics["support"],
            "coverage": round(metrics["coverage"], 6),
            "confusion": metrics["confusion"],
        },
        indent=2,
        ensure_ascii=False,
    ))

    # 可选：把原始预测保存下来，方便排查
    os.makedirs("eval_out", exist_ok=True)
    out_path = "eval_out/pubmedqa_decision_only_samples.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(df)):
            f.write(json.dumps(
                {
                    "idx": i,
                    "question": questions[i],
                    "ctx": ctx_list[i],
                    "gold_final": gold_decisions[i],
                    "gen_text": preds[i],
                    "pred_decision": pred_decisions[i],
                },
                ensure_ascii=False,
            ) + "\n")
    print(f"\nSaved samples -> {out_path}")


if __name__ == "__main__":
    main()
