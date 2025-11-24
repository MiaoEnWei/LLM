import argparse, json, torch, re, os
from transformers import AutoTokenizer, AutoModelForCausalLM

LETTER = "ABCD"

def extract_letter(txt: str) -> str:
    m = re.search(r"[ABCDabcd]", txt)
    return m.group(0).upper() if m else "?"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型或本地权重路径")
    ap.add_argument("--infile", required=True, help="data/infer/..._prompts.jsonl")
    ap.add_argument("--outfile", required=True, help="输出预测 jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 读取 prompts
    prompts = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            prompts.append((o["id"], o["prompt"]))

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    w = open(args.outfile, "w", encoding="utf-8")

    mdl.eval()
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i+args.batch_size]
        ids  = [x[0] for x in batch]
        ins  = [x[1] for x in batch]

        enc = tok(ins, return_tensors="pt", padding=True, truncation=False).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(
                **enc,
                do_sample=(args.temperature > 0.0),
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        # 取“新增的”文本
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for ex_id, prompt_text, cont in zip(ids, ins, texts):
            letter = extract_letter(cont)
            rec = {
                "id": ex_id,
                "pred_letter": letter,
                "gen_text": cont.strip()
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    w.close()
    print(f"Saved: {args.outfile}  total={len(prompts)}")

if __name__ == "__main__":
    main()
