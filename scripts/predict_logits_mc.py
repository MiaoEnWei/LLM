# scripts/predict_logits_mc.py
import argparse, json, os, torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.utils import logging
logging.set_verbosity_error()

LETTER = "ABCD"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF模型ID或本地checkpoint目录")
    ap.add_argument("--infile", required=True, help="data/infer/..._prompts.jsonl")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--dtype", default="auto", choices=["auto","fp16","bf16","fp32"])
    args = ap.parse_args()

    dtype = None
    if args.dtype == "fp16" and torch.cuda.is_available(): dtype = torch.float16
    elif args.dtype == "bf16" and torch.cuda.is_available(): dtype = torch.bfloat16
    elif args.dtype == "fp32": dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # 关键
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only
    )
    device = model.device

    # 读 prompts
    prompts = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            # 强化约束的 prompt 尾巴（可直接覆盖你生成prompts的逻辑）
            p = o["prompt"]
            if not p.rstrip().endswith("Answer: "):
                p = p.rstrip() + "\nFinal answer (A/B/C/D) only: "
            prompts.append((o["id"], p))

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    w = open(args.outfile, "w", encoding="utf-8")

    # 预先把 4 个标签各自编码（可能是 1~2 个子词）
    label_tokens = {}
    for L in LETTER:
        ids = tok.encode(L, add_special_tokens=False)
        label_tokens[L] = ids

    # 逐批处理
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i+args.batch_size]
        ids = [x[0] for x in batch]
        texts = [x[1] for x in batch]

        # 为了计算“追加标签”的条件概率，我们分别构建 4 个候选：prompt + L
        cand_inputs = []
        cand_meta = []  # (row_idx, letter, base_len, seq_len)
        # 先算每个 prompt 的 token 长度
        base_enc = tok(texts, add_special_tokens=False)
        base_lens = [len(x) for x in base_enc["input_ids"]]

        for row_idx, (base_ids, base_len) in enumerate(zip(base_enc["input_ids"], base_lens)):
            for L in LETTER:
                lab = label_tokens[L]
                seq = base_ids + lab
                cand_inputs.append(seq)
                cand_meta.append((row_idx, L, base_len, len(seq)))

        # pad 成 batch
        max_len = max(len(x) for x in cand_inputs)
        pad_id = tok.pad_token_id
        input_ids = []
        attn = []
        for seq in cand_inputs:
            pad = [pad_id] * (max_len - len(seq))
            input_ids.append(pad + seq if tok.padding_side == "left" else seq + pad)
            attn.append([0]*(max_len - len(seq)) + [1]*len(seq) if tok.padding_side == "left" else [1]*len(seq)+[0]*(max_len - len(seq)))
        input_ids = torch.tensor(input_ids, device=device)
        attention_mask = torch.tensor(attn, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logp = log_softmax(logits, dim=-1)

        # 计算每个候选的追加标签对数概率（逐 token 累加）
        # 注意：第 t 个 token 的对数概率来自 logits 的 t-1 位置
        # 因此我们对每条序列累加从 base_len 到 seq_len-1 的 token 的 logprob
        scores = [float("-inf")] * len(cand_inputs)
        offset = 0 if tok.padding_side == "left" else max_len - 0
        for j, (row_idx, L, base_len, seq_len) in enumerate(cand_meta):
            # 把真实的索引换算到张量维度
            if tok.padding_side == "left":
                start = max_len - seq_len
                base_start = start + base_len
                # 累加 labels 的 logprob：位置 base_start..(max_len-1) 的 token，
                # 其概率来自 logits 的位置 base_start-1..(max_len-2)
                s = 0.0
                for pos in range(base_start, max_len):
                    prev = pos-1
                    token_id = input_ids[j, pos].item()
                    s += logp[j, prev, token_id].item()
            else:
                base_start = base_len - 1
                s = 0.0
                for pos in range(base_len, seq_len):
                    prev = pos - 1
                    token_id = input_ids[j, pos].item()
                    s += logp[j, prev, token_id].item()
            scores[j] = s

        # 对每个原始样本选分数最高的字母
        best = {row_idx: ("A", float("-inf")) for row_idx in range(len(texts))}
        for j, (row_idx, L, _, _) in enumerate(cand_meta):
            s = scores[j]
            if s > best[row_idx][1]:
                best[row_idx] = (L, s)

        # 写出结果
        for row_idx, ex_id in enumerate(ids):
            letter, s = best[row_idx]
            w.write(json.dumps({"id": ex_id, "pred_letter": letter, "score": s}, ensure_ascii=False) + "\n")

    w.close()
    print(f"Saved: {args.outfile}  total={len(prompts)}")

if __name__ == "__main__":
    main()
