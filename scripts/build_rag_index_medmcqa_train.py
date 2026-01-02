import os
import json
import argparse
import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, default="./data/medmcqa/train.json", help="MedMCQA train.json (JSONL)")
    p.add_argument("--out_dir", type=str, default="./rag_cache/medmcqa_train", help="output cache dir")
    p.add_argument("--embed_model", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    p.add_argument("--max_docs", type=int, default=0, help="0=all, otherwise limit docs for speed (e.g., 50000)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_q_chars", type=int, default=300)
    p.add_argument("--max_opt_chars", type=int, default=180)
    return p.parse_args()

def read_jsonl(path, max_docs=0):
    docs = []
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            docs.append(ex)
            n += 1
            if max_docs and n >= max_docs:
                break
    return docs

def build_doc_text(ex, max_q_chars=300, max_opt_chars=180):
    q = (ex.get("question") or "").strip().replace("\n", " ")[:max_q_chars]
    opa = (ex.get("opa") or "").strip().replace("\n", " ")[:max_opt_chars]
    opb = (ex.get("opb") or "").strip().replace("\n", " ")[:max_opt_chars]
    opc = (ex.get("opc") or "").strip().replace("\n", " ")[:max_opt_chars]
    opd = (ex.get("opd") or "").strip().replace("\n", " ")[:max_opt_chars]

    cop = ex.get("cop", None)
    s = str(cop).strip()
    # ✅ MedMCQA 这里按 1~4
    m = {"1": "A", "2": "B", "3": "C", "4": "D"}
    ans = m.get(s)
    if ans is None:
        return None

    correct = {"A": opa, "B": opb, "C": opc, "D": opd}[ans]
    # doc = 相似题 + 正确选项文本（in-domain 最有效的形式之一）
    return f"Q: {q}\nA: {correct}"

@torch.no_grad()
def encode_texts(texts, tok, model, device, batch_size=128, max_length=128):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        inputs = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
        embs.append(cls)
    return np.vstack(embs)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading train jsonl: {args.train_file}")
    data = read_jsonl(args.train_file, max_docs=args.max_docs)

    texts = []
    kept = 0
    skipped = 0
    for ex in data:
        t = build_doc_text(ex, args.max_q_chars, args.max_opt_chars)
        if t is None:
            skipped += 1
            continue
        texts.append(t)
        kept += 1

    print(f"Docs kept: {kept}, skipped(bad cop): {skipped}")

    print("Loading embed model...")
    tok = AutoTokenizer.from_pretrained(args.embed_model)
    model = AutoModel.from_pretrained(args.embed_model).to(device)
    model.eval()

    embs = encode_texts(texts, tok, model, device, batch_size=args.batch_size, max_length=128).astype("float32")
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    docs_path = os.path.join(args.out_dir, "docs.jsonl")
    index_path = os.path.join(args.out_dir, "index.faiss")

    with open(docs_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    faiss.write_index(index, index_path)

    print("Saved:")
    print("  docs :", docs_path)
    print("  index:", index_path)

if __name__ == "__main__":
    main()
