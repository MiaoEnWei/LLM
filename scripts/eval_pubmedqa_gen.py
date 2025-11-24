# scripts/eval_pubmedqa_gen.py
"""
PubMedQA 长答案生成评测（ROUGE + 可选决策一致性ACC）
- 从本地 Parquet 读取 pqa_labeled
- 生成型评测：使用指定 Causal LM（GPT-2 / Llama-2 / -chat）
- 可选从生成文本中抽取 Yes/No/Maybe 统计决策一致性
- 支持设定随机种子，尽量保证可复现
- 支持加载 PEFT 适配器 (--adapter)，例如 Prompt Tuning / LoRA
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
import evaluate

# 尝试导入 peft（用于加载 Prompt Tuning / LoRA 适配器）
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


TEMPLATE = (
    "You are a biomedical QA assistant.\n"
    "Question: {q}\n"
    "Context: {ctx}\n\n"
    "Provide a concise rationale (1-2 sentences) and end with a final decision word among [Yes, No, Maybe].\n"
    "Answer: "
)


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
    """根据是否 chat 模型，构造 prompt。"""
    if use_chat_template and hasattr(tok, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": "You are a helpful biomedical QA assistant."},
            {
                "role": "user",
                "content": (
                    f"Question: {q}\nContext: {ctx}\n"
                    "Provide a concise rationale (1-2 sentences) and end with a final decision word among [Yes, No, Maybe].\n"
                    "Answer:"
                ),
            },
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return TEMPLATE.format(q=q, ctx=ctx)


def extract_decision(text: str):
    """
    从生成的长答案末尾尽量抽取 yes/no/maybe（大小写不敏感）。
    优先匹配句末；若失败，则从全文中取最后一次出现。
    """
    t = (text or "").strip().lower()
    m = re.search(r"\b(yes|no|maybe)\b\.?$", t)
    if m:
        return m.group(1)
    cand = re.findall(r"\b(yes|no|maybe)\b", t)
    return cand[-1] if cand else None


def decision_metrics(pred_list, gold_list):
    """给出基础 ACC 和混淆统计（不依赖 sklearn）。"""
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


def one_line(s: str, max_len: int = 300) -> str:
    """打印示例时，把换行压成一行，方便在终端看。"""
    if s is None:
        return ""
    return s[:max_len].replace("\n", " ").strip()


def load_model_and_tokenizer(model_path: str, adapter_path: str, local_files_only: bool):
    """统一加载 base 模型 + 可选 adapter（Prompt Tuning / LoRA）。"""
    print(f"[eval_pubmedqa] Loading tokenizer from {model_path}")
    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_fp16 = torch.cuda.is_available()
    load_kwargs = {
        "local_files_only": local_files_only,
    }
    if use_fp16:
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    print(f"[eval_pubmedqa] Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs,
    )

    if adapter_path:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "检测到 --adapter，但当前环境未安装 peft。\n"
                "请先运行: pip install peft"
            )
        print(f"[eval_pubmedqa] Loading PEFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=local_files_only,
        )
        print("[eval_pubmedqa] Adapter applied.")

    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="pqa_labeled 的 parquet 路径")
    ap.add_argument("--model", required=True, help="HF 模型名或本地权重目录")
    ap.add_argument("--adapter", default="", help="PEFT 适配器目录 (Prompt Tuning / LoRA)，可留空")
    ap.add_argument("--use_chat_template", action="store_true", help="对 -chat 模型启用 chat 模板")
    ap.add_argument("--limit", type=int, default=200, help="评测样本数（从头截取）")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_ctx_chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true", help="静默 Transformers 日志")
    ap.add_argument("--with_decision_acc", action="store_true", help="同时评估结论一致性ACC")
    ap.add_argument("--local_files_only", action="store_true", help="强制仅使用本地模型/分词器文件")
    args = ap.parse_args()

    if args.quiet:
        hf_logging.set_verbosity_error()

    # 随机种子
    set_seed(args.seed)

    # 读取 parquet 数据
    print(f"[eval_pubmedqa] Loading parquet from {args.parquet}")
    tbl = pq.read_table(args.parquet)
    pdf = tbl.to_pandas()

    # 只取有 question / context / long_answer 的行
    pdf = pdf[["question", "context", "long_answer"]].dropna().head(args.limit)
    pdf["ctx"] = pdf["context"].map(lambda c: join_context(c, args.max_ctx_chars))

    # 加载模型 & tokenizer & adapter
    model, tok = load_model_and_tokenizer(
        model_path=args.model,
        adapter_path=args.adapter,
        local_files_only=args.local_files_only,
    )
    device = next(model.parameters()).device
    print(f"[eval_pubmedqa] Model device: {device}")

    preds, refs = [], []

    print(f"[eval_pubmedqa] Start generation, total {len(pdf)} examples...")
    for i, r in pdf.iterrows():
        q = r["question"]
        ctx = r["ctx"]
        ref = r["long_answer"]

        prompt = build_prompt(tok, args.use_chat_template, q, ctx)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # 为了可复现，关闭采样
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen = tok.decode(gen_ids, skip_special_tokens=True).strip()

        preds.append(gen)
        refs.append(ref)

        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(pdf)}]")

    # 计算 ROUGE
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(predictions=preds, references=refs)
    rouge_res = {k: float(v) for k, v in rouge_res.items()}
    print("\n=== ROUGE (Aggregated) ===")
    print(json.dumps(rouge_res, indent=2, ensure_ascii=False))

    # -------- 打印 top5 / low5 / median5（按 rougeLsum）--------
    print("\n[eval_pubmedqa] Computing per-example rougeLsum for ranking...")
    rougeLsum_scores = []
    for pred, ref in zip(preds, refs):
        s = rouge.compute(predictions=[pred], references=[ref])["rougeLsum"]
        rougeLsum_scores.append(float(s))

    idx_sorted = sorted(range(len(rougeLsum_scores)), key=lambda i: rougeLsum_scores[i])
    n = len(idx_sorted)

    def show_examples(title, indices):
        print(f"\n[{title}]  (by rougeLsum)")
        for i in indices:
            score = rougeLsum_scores[i]
            q = pdf.iloc[i]["question"]
            print(f"- #{i:04d}  score={score:.4f}")
            print(f"  Q: {one_line(q, 200)}")
            print(f"  PRED: {one_line(preds[i], 300)}")
            print(f"  REF:  {one_line(refs[i], 300)}\n")

    top_k = 5
    low5_idx = idx_sorted[:top_k]
    top5_idx = idx_sorted[-top_k:][::-1]  # 从高到低
    mid_start = max(0, n // 2 - top_k // 2)
    mid_idx = idx_sorted[mid_start:mid_start + top_k]

    show_examples("TOP 5", top5_idx)
    show_examples("LOW 5", low5_idx)
    show_examples("MEDIAN 5", mid_idx)

    # -------- 决策一致性 (Yes / No / Maybe) --------
    if args.with_decision_acc:
        print("\n[eval_pubmedqa] Evaluating decision consistency (Yes/No/Maybe)...")
        gold_df = tbl.to_pandas()[["final_decision"]].dropna().head(args.limit)
        gold = [str(x).lower().strip() for x in gold_df["final_decision"].tolist()]
        pred_dec = [extract_decision(t) for t in preds]
        dec_res = decision_metrics(pred_dec, gold)

        print("\n=== Decision Consistency (Yes/No/Maybe) ===")
        print(
            json.dumps(
                {
                    "acc": round(dec_res["acc"], 6),
                    "support_extracted": dec_res["support"],
                    "confusion": dec_res["confusion"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    # -------- 保存样例 --------
    os.makedirs("eval_out", exist_ok=True)
    out_path = "eval_out/pqa_labeled_gen_samples.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for q, p, ref in zip(pdf["question"], preds, refs):
            f.write(
                json.dumps({"q": q, "pred": p, "ref": ref}, ensure_ascii=False)
                + "\n"
            )
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
