# build_pubmed_qa_index_pubmedbert.py
# 用 PubMedBERT 重新为 pubmed_documents.pkl 构建一个 768 维的 Faiss 索引

import pickle
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

DOC_PATH = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/PrimeKG/pubmed_documents.pkl"
OUT_INDEX_PATH = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/PrimeKG/pubmed_qa_pubmedbert.index"
ENCODER_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
BATCH_SIZE = 16
MAX_LEN = 256


def load_docs(path):
    with open(path, "rb") as f:
        docs_raw = pickle.load(f)
    docs = []
    for d in docs_raw:
        if isinstance(d, dict):
            title = d.get("title", "") or d.get("Title", "")
            abstract = d.get("abstract", "") or d.get("Abstract", "")
            txt = (str(title) + ". " + str(abstract)).strip()
        else:
            txt = str(d)
        docs.append(txt)
    return docs


@torch.inference_mode()
def encode_batch(texts, tok, model, device):
    enc = tok(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_tensors="pt",
    ).to(device)
    out = model(**enc)
    last_hidden = out.last_hidden_state  # (B, L, H)
    mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
    masked = last_hidden * mask
    sum_hidden = masked.sum(dim=1)      # (B, H)
    lengths = mask.sum(dim=1)           # (B, 1)
    lengths = torch.clamp(lengths, min=1e-6)
    mean_hidden = sum_hidden / lengths  # (B, H)
    return mean_hidden.cpu().numpy().astype("float32")  # (B, H)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading docs from:", DOC_PATH)
    docs = load_docs(DOC_PATH)
    print("Total docs:", len(docs))

    print("Loading encoder:", ENCODER_NAME)
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = AutoModel.from_pretrained(ENCODER_NAME).to(device)
    model.eval()

    all_vecs = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        vecs = encode_batch(batch, tok, model, device)
        all_vecs.append(vecs)
        print(f"Encoded {i + len(batch)}/{len(docs)} docs", end="\r")
    print()
    all_vecs = np.concatenate(all_vecs, axis=0)  # (N, H)

    dim = all_vecs.shape[1]
    print("Embedding dim =", dim)

    # 如果你想用 cosine，相当于先归一化再用内积
    faiss.normalize_L2(all_vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(all_vecs)
    print("Indexed vectors:", index.ntotal)

    faiss.write_index(index, OUT_INDEX_PATH)
    print("Saved index to:", OUT_INDEX_PATH)


if __name__ == "__main__":
    main()
