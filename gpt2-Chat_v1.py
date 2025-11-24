#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpt2-Chat.py — 最小但稳的 GPT-2 聊天式推理脚本（含防幻觉强化）
gpt2-Chat.py — 최소하지만 안정적인 GPT-2 채팅형 추론 스크립트(환각 억제 강화)

要点 / 포인트:
- 软停序列截断 / 소프트 스톱
- 两句硬裁剪 + 末尾强制收尾标点 / 두 문장 하드 클리핑 + 문장부호 강제 종결
- 套话禁用 + 论文腔拦截(可关) / 상투문구 + 논문체 차단(옵션)
- 严格上下文/重叠/置信度门控 / 엄격 컨텍스트/중복/신뢰도
- 领域上锁(可选) / 도메인 잠금(선택)
- KB 检索 + 多样本自一致性(可选) / KB 검색 + 자기일치(선택)

# ===== 新增（强固“只从 CONTEXT 作答”）=====
# - 上下文软/硬偏置: --context_bias, --context_hard_gate
# - 两句分段生成：--force_two_sentences, --sent1_prefix, --sent2_prefix
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Set, Tuple, Optional
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 可选依赖（检索/相似度）
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers.tokenization_utils_base",
)


# ------------------ Prompt 构造 ------------------
def build_prompt(system: str, question: str, context_text: str = "", kb_block: str = "") -> str:
    parts = []
    if system:
        parts.append(system.strip())
    if context_text:
        parts.append("<CONTEXT>\n" + context_text.strip() + "\n</CONTEXT>")
    if kb_block:
        parts.append("<KB_CONTEXT>\n" + kb_block.strip() + "\n</KB_CONTEXT>")
    parts.append("Q: " + question.strip())
    parts.append("A:")
    return "\n".join(parts)


# ------------------ 轻量后处理 ------------------
def soft_truncate(ans: str, stops=None) -> str:
    if stops is None:
        stops = ["\nQ:", "\nA:", "http://", "https://", " www.", "@"]
    cut_pos = len(ans)
    for s in stops:
        i = ans.find(s)
        if i != -1 and i < cut_pos:
            cut_pos = i
    return ans[:cut_pos].rstrip()


def clip_sentences(ans: str, max_sentences: int = 2) -> str:
    if max_sentences <= 0:
        return ans.strip()
    sents = re.split(r'(?<=[.!?])\s+', ans.strip())
    return " ".join(sents[:max_sentences]).strip()


def extract_numbers(text: str):
    t = re.sub(r"\b\d+\s*-\s*\d+\b", "", text)
    return sorted(set(re.findall(r"(?<!\d)[+-]?\d+(?:\.\d+)?(?!\d)", t)))


def numbers_within_context(ans: str, ctx: str) -> bool:
    if not ans:
        return True
    nums_ans = extract_numbers(ans)
    if not nums_ans:
        return True
    nums_ctx = extract_numbers(ctx or "")
    ctx_set = set(nums_ctx)
    return all(n in ctx_set for n in nums_ans)


def contains_forbidden(ans: str, phrases) -> bool:
    if not phrases:
        return False
    a = ans.lower()
    return any(p.lower() in a for p in phrases)


STOPWORDS = set("""
a an the this that those these is are was were be being been am do does did doing have has had having of on in at for to from by with without and or nor but so than then as if because while when where which who whom whose about into over under again further just only also very more most such not no yes can could should would may might must will shall
""".split())


def question_answer_overlap(question: str, answer: str) -> int:
    def toks(s):
        return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", (s or "").lower())
                if (len(w) >= 4 and w not in STOPWORDS)]
    qset = set(toks(question))
    aset = set(toks(answer))
    return len(qset.intersection(aset))


# ============== KB 检索与拼接 ==============
def read_kb_files(kb_dir: str) -> List[str]:
    docs = []
    p = Path(kb_dir)
    if not p.exists() or not p.is_dir():
        return docs
    for fp in sorted(p.iterdir()):
        if fp.is_file():
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore").strip()
                if txt:
                    docs.append(txt)
            except Exception:
                pass
    return docs


def tfidf_topk_blocks(docs: List[str], query: str, k: int, max_chars: int) -> Tuple[str, List[Tuple[int, float]]]:
    if not docs or TfidfVectorizer is None or cosine_similarity is None:
        return "", []
    vec = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
    mat = vec.fit_transform(docs)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat)[0]
    idx = sims.argsort()[::-1][:max(1, k)]
    lines, hits = [], []
    for rank, i in enumerate(idx, start=1):
        sc = float(sims[i])
        snippet = docs[i][:max_chars]
        lines.append(f"[Doc#{rank} | score={sc:.3f}]\n{snippet}\n")
        hits.append((int(i), sc))
    return ("".join(lines)).strip(), hits


def self_consistency_center(cands: List[str]) -> Tuple[str, float]:
    if len(cands) == 1 or TfidfVectorizer is None or cosine_similarity is None:
        return (cands[0] if cands else ""), 1.0
    vec = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform(cands)
    sims = cosine_similarity(X, X).mean(axis=1)
    i = int(sims.argmax())
    return cands[i], float(sims[i])


# ============== 上下文偏置(软/硬) ==============
def build_context_id_set(tok: AutoTokenizer, context: str, kb_block: str) -> Set[int]:
    text = (context or "") + "\n" + (kb_block or "")
    if not text.strip():
        return set()
    # 用 tokenizer 的词片直接统计（更鲁棒）
    ids = tok.encode(text, add_special_tokens=False)
    return set(ids)


class ContextBiasProcessor(torch.nn.Module):
    """对出现在 CONTEXT/KB 的 token 施加 +bias；可选硬门控(对非集合施加大负偏置)"""
    def __init__(self, allow_ids: Set[int], bias: float = 0.0, hard_gate: bool = False):
        super().__init__()
        self.allow = None
        if allow_ids:
            self.allow = torch.tensor(sorted(list(allow_ids)), dtype=torch.long)
        self.bias = float(bias)
        self.hard = bool(hard_gate)
        # 标点/空白等通用符号保留
        self._safe_ascii = set(map(ord, list(" .,:;!?-'\"()[]{}\n\t")))

    def forward(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.allow is None or (self.bias == 0.0 and not self.hard):
            return scores
        V = scores.size(-1)
        device = scores.device
        out = scores
        if self.hard:
            # 先全部置为 -inf，再放行 allow + ASCII 安全符号附近 id（启发式）
            mask = torch.full_like(out, float("-inf"))
            mask[:, :] = float("-inf")
            if self.allow.numel() > 0:
                mask[:, self.allow.to(device)] = 0.0
            # 粗略放行较低 id 的常见符号（GPT-2 词表中符号多在低位，不严格）
            # 仅作兜底，避免完全卡死
            # （不依赖具体词表，只起到允许少量非上下文token的作用）
            return out + mask
        else:
            if self.allow.numel() > 0 and self.bias != 0.0:
                idx = self.allow.to(device)
                out = out.clone()
                out[:, idx] += self.bias
            return out


# ------------------ 主函数 ------------------
def main():
    ap = argparse.ArgumentParser()
    # 模型与输入
    ap.add_argument("--model", default="/home/mew/mev/llm/gpt2",
                    help="HF 模型名或本地目录（gpt2 / gpt2-medium / /home/.../gpt2） | HF 저장소명 또는 로컬 경로")
    ap.add_argument("--system", default="Answer concisely and do not invent facts.",
                    help="系统指令 / 시스템 역할 프롬프트")
    ap.add_argument("--question", required=True, help="用户问题 / 사용자 질문")
    ap.add_argument("--context_file", default="",
                    help="可选：注入到 <CONTEXT> 的文件 | 선택: <CONTEXT>로 주입할 파일")

    # 解码
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--min_new_tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=5)

    # 防幻觉后处理
    ap.add_argument("--clip_sentences", type=int, default=2)
    ap.add_argument("--stop_on", default="\\nQ:,\\nA:,http://,https://, www.,@")
    ap.add_argument("--strict_context", action="store_true")
    ap.add_argument("--forbid_urls", action="store_true")
    ap.add_argument("--forbid_phrases", default=(
        "as an ai,i think,it seems,click here,read more,subscribe,follow me,"
        "the problem with this approach,in other words,for example,the best thing,"
        "can be defined,the following,the first step,the main goal of this paper,in this paper"
    ))
    ap.add_argument("--disable_paper_guard", action="store_true")

    # 置信度/相关性门控
    ap.add_argument("--min_avg_logprob", type=float, default=None)
    ap.add_argument("--require_overlap", action="store_true")
    ap.add_argument("--overlap_min_count", type=int, default=1)

    # 领域上锁（可选）
    ap.add_argument("--domain_lock", action="store_true")
    ap.add_argument("--domain_terms", default=(
        "overfit,underfit,train,model,data,dataset,generaliz,regulariz,validat,holdout,"
        "noise,signal,bias,variance,loss,error,risk,cross,entropy,grad,learn,epoch,batch,"
        "feature,label,split,augment,early stop,early-stop,reentrancy,nonreentrant,non-reentrant,mutex,guard,withdraw,external,call,fallback,receive,msg.sender,msg.value,balance,state,update,drain,attack,solidity"
    ))

    # KB 检索 + 多样本
    ap.add_argument("--kb_dir", default="")
    ap.add_argument("--evidence_topk", type=int, default=6)
    ap.add_argument("--kb_max_chars", type=int, default=1200)
    ap.add_argument("--n_samples", type=int, default=1)

    # ===== 新增：上下文偏置与两句分段 =====
    ap.add_argument("--context_bias", type=float, default=2.5,
                    help="对出现在 CONTEXT/KB 的 token 加偏置（复制倾向）。0=关闭")
    ap.add_argument("--context_hard_gate", action="store_true",
                    help="硬门控：强烈压制不在 CONTEXT/KB 的 token（谨慎使用）")
    ap.add_argument("--force_two_sentences", action="store_true",
                    help="先生成第1句到句号，再接着生成第2句到句号")
    ap.add_argument("--sent1_prefix", default="",
                    help="第一句开头前缀，如 'Reentrancy occurs when '")
    ap.add_argument("--sent2_prefix", default="",
                    help="第二句开头前缀，如 'Mitigation: '")

    # 其他
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # 读 CONTEXT
    ctx_text = ""
    if args.context_file:
        p = Path(args.context_file)
        if p.is_file():
            ctx_text = p.read_text(encoding="utf-8", errors="ignore")
        else:
            print(f"[Warn] context_file not found: {p}")

    # 读取 KB 并检索
    kb_block = ""
    kb_hits: List[Tuple[int, float]] = []
    kb_docs: List[str] = []
    if args.kb_dir:
        kb_docs = read_kb_files(args.kb_dir)
        if kb_docs and TfidfVectorizer and cosine_similarity:
            kb_block, kb_hits = tfidf_topk_blocks(kb_docs, args.question, args.evidence_topk, args.kb_max_chars)
        elif args.kb_dir and not kb_docs:
            print(f"[Warn] kb_dir is empty or invalid: {args.kb_dir}")
        elif TfidfVectorizer is None:
            print("[Warn] scikit-learn is not installed; KB retrieval disabled.")

    # 加载模型
    local = Path(args.model).exists()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=local)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=local)

    # 设备
    device = "cuda:0" if (args.device in ("auto","cuda") and torch.cuda.is_available()) else "cpu"
    mdl.to(device)

    eos_id = tok.eos_token_id
    pad_id = tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id

    # 构造提示
    prompt = build_prompt(args.system, args.question, ctx_text, kb_block)
    if args.verbose:
        print("====== PROMPT BEGIN ======")
        print(prompt)
        print("======= PROMPT END =======")

    # 编码
    def enc(s: str):
        return tok(s, return_tensors="pt").to(device)

    # 通用生成参数
    do_sample = args.temperature > 0
    gen_common = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=max(1, args.min_new_tokens),
        do_sample=do_sample,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    if do_sample:
        gen_common.update(dict(
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            top_k=args.top_k
        ))

    # ========== Logits 处理器：领域锁 + 上下文偏置 ==========
    logits_processors = []

    # 领域上锁(可选，硬白名单，谨慎)
    if args.domain_lock:
        try:
            from transformers import LogitsProcessor
            vocab = tok.get_vocab()
            id2tok = {i: t for t, i in vocab.items()}
            allow_subs = [s.strip().lower() for s in args.domain_terms.split(",") if s.strip()]
            allow_ids: Set[int] = set()
            for tid, piece in id2tok.items():
                try:
                    tok_str = tok.convert_tokens_to_string([tok.convert_ids_to_tokens(tid)])
                except Exception:
                    tok_str = piece
                s = (tok_str or "").lower()
                if any(sub in s for sub in allow_subs):
                    allow_ids.add(tid)

            class DomainLock(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    mask = torch.full_like(scores, float("-inf"))
                    if allow_ids:
                        idx = torch.tensor(list(allow_ids), device=scores.device, dtype=torch.long)
                        mask[:, idx] = 0.0
                        return scores + mask
                    return scores
            logits_processors.append(DomainLock())
        except Exception as e:
            print(f"[Warn] domain_lock disabled due to error: {e}")

    # 上下文偏置（软/硬）
    allow_ctx_ids = build_context_id_set(tok, ctx_text, kb_block)
    if allow_ctx_ids and (args.context_bias > 0.0 or args.context_hard_gate):
        proc = ContextBiasProcessor(allow_ctx_ids, bias=args.context_bias, hard_gate=args.context_hard_gate)
        class _Wrap(torch.nn.Module):
            def __call__(self, input_ids, scores):
                return proc(scores)
        logits_processors.append(_Wrap())

    # ========== 生成 ==========
    def generate_once(prefix_text: str) -> str:
        # prefix_text 会被直接拼到提示后面，用于“前缀约束”
        full_prompt = prompt + ("\n" if not prompt.endswith("\n") else "") + prefix_text
        inputs = enc(full_prompt)
        with torch.inference_mode():
            out = mdl.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                logits_processor=logits_processors if logits_processors else None,
                num_return_sequences=max(1, int(args.n_samples)),
                **gen_common
            )
        seqs = out.sequences
        prompt_len = inputs["input_ids"].shape[1]
        # 只解码新 tokens
        texts = [tok.decode(seqs[i, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                 for i in range(seqs.size(0))]
        # 多样本时取中心
        if len(texts) == 1:
            return texts[0]
        center, _ = self_consistency_center(texts)
        return center

    answer = ""

    if args.force_two_sentences:
        # 第一句
        pre1 = args.sent1_prefix or ""
        seg1 = generate_once(pre1)
        # 截到第一个句号
        s1 = seg1.split(".")[0].strip()
        if s1 and not s1.endswith("."):
            s1 += "."
        # 第二句
        pre2 = (pre1 + s1 + " " + (args.sent2_prefix or "")) if pre1 else (s1 + " " + (args.sent2_prefix or ""))
        seg2 = generate_once(pre2)
        s2 = seg2.split(".")[0].strip()
        if s2 and not s2.endswith("."):
            s2 += "."
        answer = (pre1 + s1 + " " + (args.sent2_prefix or "") + s2).strip()
    else:
        # 单段生成（不分句）
        seg = generate_once("")
        answer = seg

    # 软停
    stops = [s for s in (args.stop_on.split(",") if args.stop_on is not None else []) if s]
    if stops:
        answer = soft_truncate(answer, stops=stops)

    # 裁句
    if args.clip_sentences > 0:
        answer = clip_sentences(answer, max_sentences=args.clip_sentences)

    # 收尾标点
    if answer and answer[-1] not in (".", "!", "?", "。", "！", "？"):
        answer = answer.rstrip() + "."

    # URL/套话/论文腔/邮箱拦截
    if args.forbid_urls and (("http://" in answer) or ("https://" in answer) or (" www." in answer)):
        if args.strict_context:
            answer = "insufficient_evidence"

    forbid_list = [p.strip() for p in (args.forbid_phrases.split(",") if args.forbid_phrases else []) if p.strip()]
    if forbid_list and contains_forbidden(answer, forbid_list):
        answer = "insufficient_evidence"

    paper_pat = re.compile(
        r"\b(the main goal of this paper|in this paper|can be defined|the following|"
        r"the first (?:step|word|thing)|in other words|for example|the best thing)\b",
        re.I
    )
    if (not args.disable_paper_guard) and paper_pat.search(answer):
        answer = "insufficient_evidence"

    if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", answer, re.I):
        if args.strict_context:
            answer = "insufficient_evidence"

    if args.strict_context and (ctx_text or kb_block) and answer != "insufficient_evidence":
        merged_ctx = (ctx_text or "") + "\n" + (kb_block or "")
        if not numbers_within_context(answer, merged_ctx):
            answer = "insufficient_evidence"

    if args.require_overlap and answer != "insufficient_evidence":
        overlap = question_answer_overlap(args.question, answer)
        if overlap < max(1, args.overlap_min_count):
            answer = "insufficient_evidence"

    print("\n" + "=" * 40)
    print(answer if answer else "insufficient_evidence")
    print("=" * 40)


if __name__ == "__main__":
    main()
