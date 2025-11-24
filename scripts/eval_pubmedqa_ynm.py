import argparse, os, math, json, re
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import Dataset

LABELS = ["yes","no","maybe"]
# 多个 verbalizer 提高鲁棒性（大小写、带句点）
VERBALIZERS = {
    "yes":   ["Yes", "yes", "YES", "Yes.", "yes."],
    "no":    ["No", "no", "NO", "No.", "no."],
    "maybe": ["Maybe", "maybe", "MAYBE", "Maybe.", "maybe."],
}

TEMPLATE = (
    "You are a biomedical QA assistant.\n"
    "Question: {q}\n"
    "Context: {ctx}\n\n"
    "Answer with exactly one word from [Yes, No, Maybe].\n"
    "Final answer: "
)

def join_context_field(ctx_obj, max_chars=4000):
    # ctx_obj: dict, with "contexts": list[str]
    if isinstance(ctx_obj, dict) and "contexts" in ctx_obj and ctx_obj["contexts"]:
        txt = " ".join(ctx_obj["contexts"])
        return txt[:max_chars]
    return ""

@torch.no_grad()
def score_candidate(model, tokenizer, prompt_ids, cand_text, device):
    """
    计算 P(cand | prompt) 的对数似然。做法：拼接 prompt + cand，
    只累加 cand 对应 token 的 logprob。
    """
    # 注意：要让 cand_text前面有个空格以便英文分词更稳定
    cand = " " + cand_text
    cand_ids = tokenizer.encode(cand, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids + cand_ids], device=device)
    outputs = model(input_ids, labels=input_ids)
    # labels 会与 input_ids 对齐，loss 是平均的，我们自己取 token-wise logprob
    # 取最后 len(cand_ids) 个位置的 token 对应的交叉熵
    # cross-entropy = -logprob
    # 从 logits 取出对应概率：
    logits = outputs.logits[:, :-1, :]  # 预测下一个
    shift_labels = input_ids[:, 1:]
    # 只保留 candidate 段
    keep = shift_labels.shape[1] - len(cand_ids)
    cand_logits = logits[:, keep:, :]
    cand_labels = shift_labels[:, keep:]
    log_probs = torch.log_softmax(cand_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, cand_labels.unsqueeze(-1)).squeeze(-1)  # [1, Lc]
    return token_logprobs.sum().item()  # 标量

def pick_label_by_ll(model, tokenizer, prompt, device):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # 为避免上下文过长，截断到模型最大输入（保留结尾）
    max_len = getattr(model.config, "max_position_embeddings", 2048) - 32
    if len(prompt_ids) > max_len:
        prompt_ids = prompt_ids[-max_len:]
    scores = {}
    for lab in LABELS:
        cand_list = VERBALIZERS[lab]
        # 对同一标签的多个 verbalizer 取 logsumexp 做“口径集成”
        cand_scores = []
        for cand in cand_list:
            s = score_candidate(model, tokenizer, prompt_ids, cand, device)
            cand_scores.append(s)
        # logsumexp 做稳定合并
        m = max(cand_scores)
        scores[lab] = m + math.log(sum(math.exp(x - m) for x in cand_scores))
    # 取分数最高的标签
    pred = max(scores.items(), key=lambda x: x[1])[0]
    return pred, scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="pqa_labeled/train-00000-of-00001.parquet")
    ap.add_argument("--model", required=True, help="HF模型名或本地路径，例如 gpt2 / ./gpt2 / meta-llama/Llama-2-7b-hf（本地）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--max_ctx_chars", type=int, default=3000)
    ap.add_argument("--use_chat_template", action="store_true",
                    help="对Llama-2-chat等需要chat模板的模型启用")
    args = ap.parse_args()

    # 读取本地 labeled parquet
    table = pq.read_table(args.parquet)
    df = table.to_pandas()
    df = df[["question","context","final_decision"]].dropna()
    df = df.head(args.limit).copy()
    df["context_text"] = df["context"].map(lambda c: join_context_field(c, args.max_ctx_chars))

    # 加载模型
    print(f"Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        device_map="auto" if args.device.startswith("cuda") else None
    )
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    y_true, y_pred = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        q, ctx, y = row.question, row.context_text, str(row.final_decision).lower().strip()
        if args.use_chat_template and hasattr(tok, "apply_chat_template"):
            msgs = [
                {"role": "system", "content": "You are a helpful biomedical QA assistant."},
                {"role": "user", "content":
                 f"Question: {q}\nContext: {ctx}\nAnswer with one of [Yes, No, Maybe].\nFinal answer: "}
            ]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = TEMPLATE.format(q=q, ctx=ctx)

        pred, _ = pick_label_by_ll(model, tok, prompt, device)
        y_true.append(y if y in LABELS else "maybe")  # 保险：异常值归为 maybe
        y_pred.append(pred)

        if (i+1) % 50 == 0:
            acc = accuracy_score(y_true, y_pred)
            print(f"[{i+1}/{len(df)}] running acc={acc:.4f}")

    acc = accuracy_score(y_true, y_pred)
    print("\n== Results ==")
    print("ACC:", f"{acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))

    # 保存预测
    out = [{"question": df.iloc[i]["question"],
            "pred": y_pred[i], "gold": y_true[i]} for i in range(len(y_true))]
    os.makedirs("eval_out", exist_ok=True)
    with open("eval_out/pqa_labeled_eval.jsonl", "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("Saved -> eval_out/pqa_labeled_eval.jsonl")

if __name__ == "__main__":
    main()
