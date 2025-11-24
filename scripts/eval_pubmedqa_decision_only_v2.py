"""
PubMedQA decision-only evaluation (Yes / No / Maybe), v2

不用正则从生成文本里抠答案，而是直接：
- 给定 prompt
- 看下一个 token 是 "Yes" / "No" / "Maybe" 三个里哪个概率最大
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def join_context(c, max_chars: int) -> str:
    """兼容 dict/list/np.ndarray 的 context 展平"""
    import numpy as np

    if c is None:
        return ""
    if isinstance(c, dict):
        ctxs = c.get("contexts", [])
    elif isinstance(c, np.ndarray):
        ctxs = c.tolist()
    elif isinstance(c, (list, tuple)):
        ctxs = c
    else:
        return str(c)[:max_chars]

    if len(ctxs) == 0:
        return ""
    out = " ".join(str(x) for x in ctxs)
    return out[:max_chars]


def normalize_decision(x: str) -> str:
    s = (str(x) or "").strip().lower()
    if s.startswith("y"):
        return "yes"
    if s.startswith("n"):
        return "no"
    return "maybe"


def build_prompt(q: str, ctx: str) -> str:
    return (
        "You are a biomedical QA assistant.\n"
        f"Question: {q}\n"
        f"Context: {ctx}\n\n"
        "Answer with only one word among [Yes, No, Maybe].\n"
        "Answer:"
    )


def decision_metrics(pred_list, gold_list):
    assert len(pred_list) == len(gold_list)
    valid_idx = [i for i, p in enumerate(pred_list) if p is not None]
    if not valid_idx:
        return {"acc": 0.0, "support": 0, "confusion": {}}

    preds = [pred_list[i] for i in valid_idx]
    golds = [gold_list[i] for i in valid_idx]

    correct = sum(int(p == g) for p, g in zip(preds, golds))
    acc = correct / len(valid_idx)

    labels = ["yes", "no", "maybe"]
    conf = Counter((g, p) for g, p in zip(golds, preds))
    confusion = {f"{g}->{p}": conf[(g, p)] for g in labels for p in labels}

    return {"acc": acc, "support": len(valid_idx), "confusion": confusion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="pqaa_labeled_test.parquet 路径")
    ap.add_argument("--model", required=True, help="基座模型路径，如 ./gpt2 或 ./llama2")
    ap.add_argument("--adapter", default="", help="PEFT 适配器目录，可选")
    ap.add_argument("--limit", type=int, default=500, help="评测样本数")
    ap.add_argument("--max_ctx_chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        hf_logging.set_verbosity_error()

    set_seed(args.seed)

    # ---- 数据 ----
    tbl = pq.read_table(args.parquet)
    df = tbl.to_pandas()

    needed = ["question", "context", "final_decision"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"parquet 缺少列 {col}, 当前列: {list(df.columns)}")

    df = df[needed].dropna().head(args.limit)
    questions = df["question"].tolist()
    ctx_list = [join_context(c, args.max_ctx_chars) for c in df["context"].tolist()]
    gold_decisions = [normalize_decision(x) for x in df["final_decision"].tolist()]

    # ---- 模型 ----
    print(f"[DecisionEval-v2] Loading tokenizer from: {args.model}")
    tok = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=args.local_files_only
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[DecisionEval-v2] Loading base model from: {args.model}")
    use_fp16 = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if use_fp16 else torch.float32,
        device_map="auto" if use_fp16 else None,
        local_files_only=args.local_files_only,
    )
    model.config.pad_token_id = tok.pad_token_id

    if args.adapter:
        if not PEFT_AVAILABLE:
            raise ImportError("指定了 --adapter，但当前环境未安装 peft")
        print(f"[DecisionEval-v2] Loading PEFT adapter from: {args.adapter}")
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            local_files_only=args.local_files_only,
        )

    model.eval()

    # 预先算出三个 label 的 token id（注意前面加空格）
    label_tokens = {}
    for lab in ["yes", "no", "maybe"]:
        # 对 GPT-2/llama 都习惯用前导空格
        ids = tok(" " + lab.capitalize(), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            # 如果被切成多个 token，就取第一个，简单近似
            label_tokens[lab] = ids[0]
        else:
            label_tokens[lab] = ids[0]

    preds = []
    all_scores = []

    # ---- 逐样本打分 ----
    for i, (q, ctx) in enumerate(zip(questions, ctx_list)):
        prompt = build_prompt(q, ctx)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :]  # 最后一个位置

        scores = {}
        for lab, tid in label_tokens.items():
            scores[lab] = float(logits[tid].item())
        # 选 logit 最大的 label
        pred_lab = max(scores.items(), key=lambda x: x[1])[0]
        preds.append(pred_lab)
        all_scores.append(scores)

        if (i + 1) % 20 == 0:
            print(f"[DecisionEval-v2] {i+1}/{len(df)}")

    # ---- 计算指标 ----
    dec_res = decision_metrics(preds, gold_decisions)

    print("\n=== PubMedQA Decision-only Evaluation v2 (logit 3-class) ===")
    print(
        json.dumps(
            {
                "acc": round(dec_res["acc"], 6),
                "support": dec_res["support"],
                "confusion": dec_res["confusion"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    # ---- 可选：保存详细预测 ----
    os.makedirs("eval_out", exist_ok=True)
    out_path = "eval_out/pqaa_decision_only_logits_preds.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (q, ctx, g, p, sc) in enumerate(
            zip(questions, ctx_list, gold_decisions, preds, all_scores)
        ):
            f.write(
                json.dumps(
                    {
                        "idx": i,
                        "question": q,
                        "context": ctx,
                        "gold": g,
                        "pred": p,
                        "scores": sc,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"\nSaved detailed preds to {out_path}")


if __name__ == "__main__":
    main()
