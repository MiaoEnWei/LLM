# scripts/eval_raft_mcqa_logits.py
import os
import json
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset


# -------------------------
# Gold label mapping
# -------------------------
def parse_gold(cop) -> Optional[str]:
    """
    MedMCQA common: cop in {"1","2","3","4"} meaning A,B,C,D
    Also supports 0-based and letters.
    """
    if cop is None:
        return None

    if isinstance(cop, int):
        if 1 <= cop <= 4:
            return ["A", "B", "C", "D"][cop - 1]
        if 0 <= cop <= 3:
            return ["A", "B", "C", "D"][cop]

    s = str(cop).strip()
    if not s:
        return None
    s_low = s.lower()

    # prioritize 1-based for string (MedMCQA dev.json often has 1..4 as strings)
    map_1based = {"1": "A", "2": "B", "3": "C", "4": "D"}
    if s in map_1based:
        return map_1based[s]

    map_other = {
        "0": "A", "1": "B", "2": "C", "3": "D",  # 0-based fallback
        "a": "A", "b": "B", "c": "C", "d": "D",
        "opa": "A", "opb": "B", "opc": "C", "opd": "D",
        "A": "A", "B": "B", "C": "C", "D": "D",
    }
    return map_other.get(s) or map_other.get(s_low)


# -------------------------
# Prompt (Context LAST)
# -------------------------
def format_prompt_letter(ex: dict, context: str) -> str:
    """
    Put Context near the end so that left-truncation keeps it.
    End with "Answer: " (trailing space) for stable tokenization.
    """
    q = ex.get("question", "")
    opa = ex.get("opa", "")
    opb = ex.get("opb", "")
    opc = ex.get("opc", "")
    opd = ex.get("opd", "")

    if context:
        context_block = f"References:\n{context}\n"
    else:
        context_block = ""

    return (
        f"Question: {q}\n"
        f"A) {opa}\n"
        f"B) {opb}\n"
        f"C) {opc}\n"
        f"D) {opd}\n"
        f"{context_block}"
        f"Answer: "
    )


# -------------------------
# Robust candidate logprob (multi-token safe)
# -------------------------
@torch.no_grad()
def seq_logprob(model, tok, prompt: str, continuation: str, device: str, max_length: int = 1024) -> float:
    """
    log P(continuation | prompt) using teacher forcing.
    continuation may tokenize into multiple tokens.
    We truncate prompt from the LEFT to fit (keep tail, where context sits).
    """
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    c_ids = tok(continuation, add_special_tokens=False).input_ids
    if len(c_ids) == 0:
        return -1e9

    max_prompt = max_length - len(c_ids)
    if max_prompt < 8:
        return -1e9

    if len(p_ids) > max_prompt:
        p_ids = p_ids[-max_prompt:]  # keep tail

    input_ids = torch.tensor([p_ids + c_ids], device=device)
    attn = torch.ones_like(input_ids, device=device)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits
    logp = torch.log_softmax(logits, dim=-1)

    start = len(p_ids)
    total = 0.0
    # token at position t uses logits at position t-1
    for j in range(len(c_ids)):
        pos = start + j - 1
        tgt = input_ids[0, start + j].item()
        total += float(logp[0, pos, tgt].item())
    return total


@torch.no_grad()
def predict_letter(model, tok, prompt: str, device: str, max_length: int = 1024) -> str:
    letters = ["A", "B", "C", "D"]
    # prompt already ends with a space after "Answer: "
    scores = [seq_logprob(model, tok, prompt, L, device=device, max_length=max_length) for L in letters]
    return letters[int(np.argmax(scores))]


# -------------------------
# RAG (cached KB)
# -------------------------
def load_cached_kb(kb_dir: str) -> Tuple[List[str], faiss.Index]:
    docs_path = os.path.join(kb_dir, "docs.jsonl")
    index_path = os.path.join(kb_dir, "index.faiss")
    if not (os.path.exists(docs_path) and os.path.exists(index_path)):
        raise FileNotFoundError(f"KB cache not found: {kb_dir} (need docs.jsonl + index.faiss)")

    docs = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # build_rag_index_medmcqa_train.py saved {"text": ...}
            docs.append(obj["text"])
    index = faiss.read_index(index_path)
    return docs, index


@torch.no_grad()
def embed_query(text: str, embed_tok, embed_model, device: str, max_length: int = 128) -> np.ndarray:
    inputs = embed_tok([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    out = embed_model(**inputs)
    vec = out.last_hidden_state[:, 0, :].detach().cpu().numpy().astype("float32")
    faiss.normalize_L2(vec)
    return vec


def retrieve_context(
    question: str,
    rag_docs: List[str],
    rag_index: faiss.Index,
    embed_tok,
    embed_model,
    device: str,
    k: int = 2,
    ctx_max_chars: int = 300,
) -> str:
    qv = embed_query(question, embed_tok, embed_model, device=device)
    _, I = rag_index.search(qv, k)

    parts = []
    for idx in I[0]:
        if idx >= 0:
            parts.append(rag_docs[idx])

    ctx = "\n---\n".join(parts)
    return ctx[:ctx_max_chars]


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--raft_model", type=str, required=True)

    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--rag_k", type=int, default=2)
    parser.add_argument("--kb_dir", type=str, default="", help="cached KB dir (docs.jsonl + index.faiss). If empty -> MedQuad fallback")
    parser.add_argument("--embed_model", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--ctx_max_chars", type=int, default=300, help="truncate retrieved context to N chars")

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=1024, help="LM max length for scoring")
    parser.add_argument("--out_jsonl", type=str, required=True)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"DEV_FILE      : {args.dev_file}")
    print(f"BASE_MODEL    : {args.base_model}")
    print(f"RAFT_MODEL    : {args.raft_model}")
    print(f"DEVICE        : {device}")
    print(f"RAG           : {'ON' if args.use_rag else 'OFF'} (k={args.rag_k})")
    if args.use_rag:
        print(f"KB_DIR        : {args.kb_dir if args.kb_dir else '(MedQuad fallback)'}")
        print(f"EMBED_MODEL   : {args.embed_model}")
        print(f"CTX_MAX_CHARS : {args.ctx_max_chars}")
    print("=" * 60)

    # load dev (JSONL)
    examples = []
    with open(args.dev_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if args.limit and len(examples) >= args.limit:
                break
    print(f"Loaded dev examples: {len(examples)}")

    # load tokenizers
    base_tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=os.path.isdir(args.base_model))
    raft_tok = AutoTokenizer.from_pretrained(args.raft_model, local_files_only=os.path.isdir(args.raft_model))
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    if raft_tok.pad_token is None:
        raft_tok.pad_token = raft_tok.eos_token

    # load LMs
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=os.path.isdir(args.base_model),
        dtype=dtype,  # avoids torch_dtype deprecation warning
    ).to(device)
    raft = AutoModelForCausalLM.from_pretrained(
        args.raft_model,
        local_files_only=os.path.isdir(args.raft_model),
        dtype=dtype,
    ).to(device)
    base.eval()
    raft.eval()

    # RAG init
    rag_docs = None
    rag_index = None
    embed_tok = None
    embed_model = None

    if args.use_rag:
        if args.kb_dir and args.kb_dir.strip():
            print("Loading cached KB index...")
            rag_docs, rag_index = load_cached_kb(args.kb_dir.strip())
            print(f"KB size: {len(rag_docs)}")
        else:
            # fallback (slow)
            print("Loading MedQuad KB (fallback)...")
            kb = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
            rag_docs = []
            for item in kb:
                q = (item.get("Question", "") or "").strip()
                a = (item.get("Answer", "") or "").strip()
                if len(a) > 20:
                    rag_docs.append(f"Q: {q}\nA: {a[:400]}")
            print(f"KB size: {len(rag_docs)}")

            print("Building MedQuad index (fallback, slow)...")
            embed_tok = AutoTokenizer.from_pretrained(args.embed_model)
            embed_model = AutoModel.from_pretrained(args.embed_model).to(device)
            embed_model.eval()

            all_vecs = []
            bs = 128
            for i in tqdm(range(0, len(rag_docs), bs), desc="Encoding KB"):
                batch = rag_docs[i:i + bs]
                inputs = embed_tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                out = embed_model(**inputs)
                vec = out.last_hidden_state[:, 0, :].detach().cpu().numpy().astype("float32")
                all_vecs.append(vec)
            doc_embs = np.vstack(all_vecs)
            faiss.normalize_L2(doc_embs)
            rag_index = faiss.IndexFlatIP(doc_embs.shape[1])
            rag_index.add(doc_embs)

        # need embed model for query encoding even if cached KB is used
        if embed_tok is None:
            embed_tok = AutoTokenizer.from_pretrained(args.embed_model)
        if embed_model is None:
            embed_model = AutoModel.from_pretrained(args.embed_model).to(device)
            embed_model.eval()

        print("RAG ready.")

    # Evaluate
    total = 0
    bad_labels = 0
    corr_base_no = 0
    corr_raft_no = 0
    corr_base_rg = 0
    corr_raft_rg = 0

    out_dir = os.path.dirname(args.out_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(tqdm(examples, desc="Evaluating")):
            gold = parse_gold(ex.get("cop"))
            if gold is None:
                bad_labels += 1
                continue

            # context
            ctx = ""
            if args.use_rag:
                ctx = retrieve_context(
                    question=ex.get("question", ""),
                    rag_docs=rag_docs,
                    rag_index=rag_index,
                    embed_tok=embed_tok,
                    embed_model=embed_model,
                    device=device,
                    k=args.rag_k,
                    ctx_max_chars=args.ctx_max_chars,
                )

            # no-rag prompt
            prompt_no = format_prompt_letter(ex, context="")
            pred_base_no = predict_letter(base, base_tok, prompt_no, device=device, max_length=args.max_length)
            pred_raft_no = predict_letter(raft, raft_tok, prompt_no, device=device, max_length=args.max_length)

            ok_base_no = (pred_base_no == gold)
            ok_raft_no = (pred_raft_no == gold)
            corr_base_no += int(ok_base_no)
            corr_raft_no += int(ok_raft_no)

            record = {
                "id": i,
                "gold": gold,
                "base_no_rag": pred_base_no,
                "raft_no_rag": pred_raft_no,
                "base_no_rag_correct": ok_base_no,
                "raft_no_rag_correct": ok_raft_no,
            }

            # rag prompt
            if args.use_rag:
                prompt_rg = format_prompt_letter(ex, context=ctx)
                pred_base_rg = predict_letter(base, base_tok, prompt_rg, device=device, max_length=args.max_length)
                pred_raft_rg = predict_letter(raft, raft_tok, prompt_rg, device=device, max_length=args.max_length)

                ok_base_rg = (pred_base_rg == gold)
                ok_raft_rg = (pred_raft_rg == gold)
                corr_base_rg += int(ok_base_rg)
                corr_raft_rg += int(ok_raft_rg)

                record.update({
                    "base_rag": pred_base_rg,
                    "raft_rag": pred_raft_rg,
                    "base_rag_correct": ok_base_rg,
                    "raft_rag_correct": ok_raft_rg,
                })

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

    def acc(c, n): return (c / n) if n else 0.0

    print("\n" + "=" * 60)
    print(f"base + no_rag | acc = {acc(corr_base_no, total):.4f} ({corr_base_no}/{total}), bad_labels={bad_labels}")
    print(f"raft + no_rag | acc = {acc(corr_raft_no, total):.4f} ({corr_raft_no}/{total}), bad_labels={bad_labels}")
    if args.use_rag:
        print(f"base + rag    | acc = {acc(corr_base_rg, total):.4f} ({corr_base_rg}/{total}), bad_labels={bad_labels}")
        print(f"raft + rag    | acc = {acc(corr_raft_rg, total):.4f} ({corr_raft_rg}/{total}), bad_labels={bad_labels}")
    print(f"Saved jsonl: {args.out_jsonl}")
    print("=" * 60)


if __name__ == "__main__":
    main()
