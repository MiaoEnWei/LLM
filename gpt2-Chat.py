#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG + Constrained Decoding + Post QA Checks (clean, enhanced)
- RAG: BM25-retrieve paragraphs -> assemble <CONTEXT>
- Decoding constraints:
  * must_include -> force_words_ids (do NOT mix with DomainLock)
  * Optional DomainLock (LogitsProcessor): enabled only when explicitly providing --domain_lock_terms
    or enabling --domain_lock_from_context
- Post-generation QA checks:
  * Keyword overlap (optional)
  * Number consistency (optional)
  * Blacklist forbid_phrases
  * Select best by sequences_scores (pick highest score when generating multiple candidates)
- Sampling / diversity:
  * Sampling: temperature/top_p/top_k + n_samples
  * If must_include is used, switch to beam search; optional diverse beam
"""

import os, re, sys, glob, argparse, warnings
from pathlib import Path
from typing import List, Tuple, Optional, Set

warnings.filterwarnings("ignore")

# ---------- Tokenization & NLTK fallback ----------
def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    try:
        import nltk
        try:
            from nltk.tokenize import word_tokenize
            return nltk.word_tokenize(text)
        except Exception:
            pass
    except Exception:
        pass
    # Fallback: regex tokenization
    return re.findall(r"[A-Za-z0-9_]+", text)

# ---------- Read & chunk ----------
def read_corpus_files(corpus_dir: str, exts=(".txt", ".md")) -> List[Tuple[str, str]]:
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(corpus_dir, f"**/*{ext}"), recursive=True)
    out = []
    for fp in files:
        try:
            txt = Path(fp).read_text(encoding="utf-8", errors="ignore")
            out.append((fp, txt))
        except Exception:
            pass
    return out

def simple_chunks(text: str, max_len=800) -> List[str]:
    paras = re.split(r"\n\s*\n+", (text or "").strip())
    chunks, buf, cur = [], [], 0
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if cur + len(p) > max_len and buf:
            chunks.append("\n".join(buf)); buf, cur = [], 0
        buf.append(p); cur += len(p)
    if buf: chunks.append("\n".join(buf))
    final = []
    for c in chunks:
        if len(c) <= max_len:
            final.append(c)
        else:
            # Sentence-level split fallback
            try:
                import nltk
                sents = nltk.tokenize.sent_tokenize(c)
            except Exception:
                sents = re.split(r'(?<=[.!?。！？])\s+', c)
            tmp, cur = [], 0
            for s in sents:
                if cur + len(s) > max_len and tmp:
                    final.append(" ".join(tmp)); tmp, cur = [], 0
                tmp.append(s); cur += len(s)
            if tmp: final.append(" ".join(tmp))
    return final

# ---------- BM25 ----------
def build_bm25(chunks: List[str]):
    from rank_bm25 import BM25Okapi
    tokenized = [_tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized, chunks

def retrieve(bm25, tokenized, chunks, query: str, topk=5):
    qtok = _tokenize(query)
    scores = bm25.get_scores(qtok)
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    return [(chunks[i], float(scores[i])) for i in idxs]

# ---------- LogitsProcessor: DomainLock ----------
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    LogitsProcessor
)

class DomainLock(LogitsProcessor):
    """Restrict the vocabulary to allowed substrings (domain stems) + common symbols."""
    def __init__(self, tok, allow_subs: List[str]):
        self.allow_subs = [s.strip().lower() for s in allow_subs if s.strip()]
        vocab = tok.get_vocab()
        id2tok = {i: t for t, i in vocab.items()}
        self.allow_ids: Set[int] = set()
        # Remove empty strings; keep common punctuation/space/newline
        extra_ok = {" ", ".", ",", ":", ";", "(", ")", "-", "_", "\n", "\"", "'", "%"}
        for tid, piece in id2tok.items():
            try:
                tok_piece = tok.convert_ids_to_tokens(tid)
                tok_str = tok.convert_tokens_to_string([tok_piece])
            except Exception:
                tok_str = piece
            s = (tok_str or "").lower()
            if any(sub in s for sub in self.allow_subs) or (tok_str in extra_ok):
                self.allow_ids.add(tid)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.allow_ids:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        idx = torch.tensor(list(self.allow_ids), device=scores.device, dtype=torch.long)
        mask[:, idx] = 0.0
        return scores + mask

# ---------- QA checks ----------
STOPWORDS = set("""
a an the this that those these is are was were be being been am do does did doing have has had having of on in at for to from by with without and or nor but so than then as if because while when where which who whom whose about into over under again further just only also very more most such not no yes can could should would may might must will shall
""".split())

def kw_overlap(q: str, a: str) -> int:
    def toks(s):
        return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", (s or "").lower())
                if (len(w) >= 4 and w not in STOPWORDS)]
    return len(set(toks(q)).intersection(set(toks(a))))

def extract_numbers(text: str) -> List[str]:
    t = re.sub(r"\b\d+\s*-\s*\d+\b", "", text or "")
    return sorted(set(re.findall(r"(?<!\d)[+-]?\d+(?:\.\d+)?(?!\d)", t)))

def numbers_within_context(ans: str, ctx: str) -> bool:
    if not ans: return True
    nums_ans = extract_numbers(ans)
    if not nums_ans: return True
    nums_ctx = set(extract_numbers(ctx or ""))
    return all(n in nums_ctx for n in nums_ans)

# ---------- Build force_words ----------
def build_force_words(tok, terms: List[str]):
    out = []
    for t in terms:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if ids: out.append(ids)
    return out

# ---------- Main pipeline ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--answer_prefix", default="")
    ap.add_argument("--system", default="Answer concisely based only on <CONTEXT>. No guesses.")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ctx_token_budget", type=int, default=400)
    ap.add_argument("--must_include", default="")
    ap.add_argument("--domain_lock_terms", default="")
    ap.add_argument("--domain_lock_from_context", action="store_true",
                    help="Build DomainLock allowlist automatically from <CONTEXT> tokens.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=5)
    ap.add_argument("--min_avg_logprob", type=float, default=None)
    ap.add_argument("--overlap_min", type=int, default=0)
    ap.add_argument("--skip_number_check", action="store_true")
    ap.add_argument("--forbid_phrases", default="")
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--diverse_beam_groups", type=int, default=1)
    ap.add_argument("--diversity_penalty", type=float, default=0.0)
    ap.add_argument("--clip_to_must", action="store_true",
                    help="Clip final answer to 'answer_prefix + must_include + .' if must phrase is found.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # 1) Read + chunk + index
    files = read_corpus_files(args.corpus_dir)
    raw_chunks = []
    for fp, txt in files:
        for ch in simple_chunks(txt, max_len=800):
            raw_chunks.append(ch)
    if not raw_chunks:
        raise SystemExit("[Error] No chunks from corpus.")
    bm25, tokenized, chunks = build_bm25(raw_chunks)

    # 2) Retrieve
    hits = retrieve(bm25, tokenized, chunks, args.question, topk=args.topk)
    ctx = "\n\n".join([c for c,_ in hits])

    # 3) Model
    local = Path(args.model).exists()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=local)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=local)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    device = "cuda:0" if ((args.device=="cuda" or args.device=="auto") and torch.cuda.is_available()) else ("cpu" if args.device!="auto" else "cpu")
    mdl.to(device)

    # 4) Compress CONTEXT
    def truncate_to_tokens(s: str, budget: int) -> str:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget: return s
        return tok.decode(ids[:budget], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ctx_use = truncate_to_tokens(ctx, args.ctx_token_budget)

    # 5) Prompt
    prompt = f"""{args.system.strip()}

<CONTEXT>
{ctx_use.strip()}
</CONTEXT>

Q: {args.question.strip()}
A: {args.answer_prefix}"""
    if args.verbose:
        print("====== PROMPT BEGIN ======"); print(prompt); print("======= PROMPT END =======")

    inputs = tok(prompt, return_tensors="pt").to(device)

    # 6) Constraints: force_words (must_include) + optional DomainLock
    must_terms = [s.strip() for s in args.must_include.split(",") if s.strip()]
    force_words_ids = build_force_words(tok, must_terms) if must_terms else None

    allow_terms = []
    if args.domain_lock_from_context:
        allow_terms = sorted(set(_tokenize(ctx_use)))
    else:
        allow_terms = [s.strip() for s in args.domain_lock_terms.split(",") if s.strip()]
    # Do NOT merge must_terms into allow_terms; that could lock the vocabulary too tightly
    logits_processors = None
    if allow_terms:
        logits_processors = [DomainLock(tok, allow_terms)]

    # 7) Generation parameters (sampling/beam & diversity)
    do_sample = (args.temperature > 0.0) and (force_words_ids is None)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(args.temperature, 1e-5)
        gen_kwargs["top_p"] = args.top_p
        if args.top_k and args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k
        if args.n_samples and args.n_samples > 1:
            gen_kwargs["num_return_sequences"] = args.n_samples

    # Forced phrase -> beam search (no sampling); supports multiple returns + diverse beam
    if force_words_ids:
        gen_kwargs.update(dict(
            num_beams=max(4, args.n_samples*2),
            num_return_sequences=max(1, args.n_samples),
            do_sample=False,
            force_words_ids=force_words_ids,
            early_stopping=True,
        ))
        if args.diverse_beam_groups and args.diverse_beam_groups > 1:
            gen_kwargs.update(dict(
                num_beam_groups=args.diverse_beam_groups,
                diversity_penalty=args.diversity_penalty
            ))

    if logits_processors:
        gen_kwargs["logits_processor"] = logits_processors

    out = mdl.generate(**inputs, **gen_kwargs)

    # 8) Extract candidates & select best
    seqs = out.sequences
    # If sequences_scores exists, pick the best (higher is better); otherwise pick index 0
    pick_idx = 0
    if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
        try:
            pick_idx = int(torch.argmax(out.sequences_scores).item())
        except Exception:
            pick_idx = 0

    seq = seqs[pick_idx]
    text = tok.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    pos = text.rfind("\nA:")
    answer = text[pos+3:].strip() if pos != -1 else text.strip()

    # --- Optional: clip to must phrase ---
    if args.clip_to_must and args.must_include and args.answer_prefix:
        must = args.must_include.split(",")[0].strip()
        if must:
            al = answer.lower()
            if must.lower() in al:
                answer = f"{args.answer_prefix}{must}."

    # --- Cleanup & two-sentence clipping ---
    def _sanitize(txt: str) -> str:
        txt = re.sub(r'0x[0-9A-Fa-f]+', '', txt or '')     # drop hex-like
        txt = re.sub(r'[—–-]{3,}', ' ', txt)               # long dashes
        txt = re.sub(r'_{2,}', ' ', txt)                   # underscores
        txt = re.sub(r'\s+', ' ', txt).strip()
        return txt

    def _clip_two_sentences(txt: str) -> str:
        txt = (txt or "").strip().replace("\n"," ")
        txt = re.sub(r"\s+", " ", txt)
        parts = re.split(r'(?<=[.!?。！？])\s+', txt)
        out = " ".join(parts[:2]).strip()
        if out and out[-1] not in ".!?。！？":
            out += "."
        return out

    answer = _clip_two_sentences(_sanitize(answer))

    # 9) Post-generation QA checks
    reasons = []
    forbids = [p.strip().lower() for p in args.forbid_phrases.split(",") if p.strip()]
    al = (answer or "").lower()
    if any(b in al for b in forbids):
        reasons.append("forbid_phrases")
    if kw_overlap(args.question, answer) < args.overlap_min:
        reasons.append(f"overlap<{max(1,args.overlap_min)}")
    if (not args.skip_number_check) and (not numbers_within_context(answer, ctx_use)):
        reasons.append("numbers_outside_context")

    ok = (len(reasons) == 0)
    print("\n" + "="*40)
    print(answer if ok and answer else "insufficient_evidence")
    print("="*40)
    if not ok:
        print("[WHY]", ", ".join(reasons), file=sys.stderr)

if __name__ == "__main__":
    main()