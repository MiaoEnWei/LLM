#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Llama-2-7B-Chat Local Inference · Enhanced Edition (12GB VRAM-friendly)
Llama-2-7B-Chat 로컬 추론 · 강화판 (12GB VRAM 친화)

- Modes: auto / fp16_gpu / int8 / fp16_offload / cpu
  모드: auto / fp16_gpu / int8 / fp16_offload / cpu
- Move ONLY "input tensors" to GPU; NEVER .to(...) the whole model (avoid 4/8bit errors)
  입력 텐서만 GPU로 이동, 모델 전체 .to(...) 금지
- int8 branch uses "module-level device_map"; if needed, place lm_head on CPU to avoid bnb .to conflicts
  int8 분기: 모듈 단위 device_map, 필요 시 lm_head를 CPU로 배치
- Supports streaming output (--stream), conservative low-temperature sampling, JSON-friendly output
  스트리밍 출력, 저온도 보수 샘플링 지원

# ====== New: Lightweight anti-hallucination mechanism / 신규: 경량 환각 방지 메커니즘 ======
# - --kb_dir points to a local knowledge-base directory (plain text: txt/md)
# - Retrieve Top-K evidence and splice into [CONTEXT], guiding the model to "retrieve evidence first, then answer"
# - Generate multiple samples; self-consistency voting + consistency score + approximate entropy -> decide (answer / refuse)
# - No fine-tuning, no internet; fully black-box external control
"""

import os, argparse, threading, gc, json, math
from pathlib import Path
import psutil, torch

# ↓ Reduce VRAM fragmentation (must be set before import torch) / GPU 메모리 파편화 완화
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig,
    TextIteratorStreamer
)
from transformers.utils import logging as hf_logging
import logging

# ====== New dependency: TF-IDF retrieval / 신규 의존성: TF-IDF 검색 ======
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ↓ Quiet by default; enable with --verbose / 기본은 조용, --verbose로 상세 로그
hf_logging.set_verbosity_error()
logging.getLogger("accelerate").setLevel(logging.ERROR)


# ========== Mode selection / 모드 선택 ==========
def pick_mode(requested: str) -> str:
    """Choose a mode based on request and hardware / 요청 + 하드웨어로 모드 선택"""
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 16:
        return "fp16_gpu"
    elif vram_gb >= 10:
        return "int8"           # 10~16GB -> int8 is the most stable / 가장 안정
    else:
        return "fp16_offload"   # Smaller VRAM -> FP16 + offload / FP16+오프로딩


def _cleanup_cuda():
    """Release VRAM held by the current process / 현재 프로세스의 VRAM 해제"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


# ========== Build model / 모델 로드 ==========
def build_model(model_dir: str, mode: str, verbose: bool=False):
    model_dir = str(Path(model_dir).expanduser().resolve())
    assert Path(model_dir).is_dir(), f"Model directory does not exist: {model_dir} / 모델 디렉터리가 없습니다."

    _cleanup_cuda()  # Clear leftover VRAM from prior failures / 이전 실패 잔여 메모리 정리

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    attn_impl = "sdpa"

    # Read number of layers / 레이어 수
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    n_layers = getattr(cfg, "num_hidden_layers", 32)

    if mode == "int8":
        # 8-bit quantization (module-level multi-device mapping)
        # 8bit 양자화(모듈 단위 다중 디바이스 매핑)
        bnb = BitsAndBytesConfig(load_in_8bit=True)

        # Put most modules on GPU0; lm_head on CPU (small and safe) to avoid whole-model .to(...)
        # 대부분 GPU0, lm_head는 CPU로 배치하여 전체 .to(...) 회피
        device_map = {"model.embed_tokens": 0, "model.norm": 0, "lm_head": "cpu"}
        device_map.update({f"model.layers.{i}": 0 for i in range(n_layers)})

        offload_dir = os.path.expanduser("~/hf_offload_int8")
        os.makedirs(offload_dir, exist_ok=True)

        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                model_dir,
                quantization_config=bnb,
                device_map=device_map,      # Key: module-level device_map / 핵심
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
                local_files_only=True,
                offload_folder=offload_dir,
            )
            if verbose:
                print(">> device_map:", getattr(mdl, "hf_device_map", None))
        except Exception as e:
            # If it still fails, fall back to FP16 + offload (more conservative)
            # 여전히 에러면 FP16+오프로딩 폴백
            if verbose:
                print(">> INT8 failed, fallback to fp16_offload. reason:", repr(e))
            ram_gb = psutil.virtual_memory().total // (1024 ** 3)
            max_mem = {0: "9.5GiB", "cpu": f"{int(ram_gb * 0.8)}GiB"}
            mdl = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_mem,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
                local_files_only=True,
            )
            if verbose:
                print(">> FALLBACK device_map:", getattr(mdl, "hf_device_map", None))

    elif mode == "fp16_gpu":
        mdl = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map={"": 0},   # All on GPU0 / 전부 GPU0
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
            local_files_only=True,
        )
        if verbose:
            print(">> device_map:", getattr(mdl, "hf_device_map", None))

    elif mode == "fp16_offload":
        ram_gb = psutil.virtual_memory().total // (1024 ** 3)
        max_mem = {0: "9.5GiB", "cpu": f"{int(ram_gb * 0.8)}GiB"}  # More conservative / 보수적
        mdl = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",              # Auto offload to CPU if it doesn't fit / 자동 오프로딩
            max_memory=max_mem,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
            local_files_only=True,
        )
        if verbose:
            print(">> device_map:", getattr(mdl, "hf_device_map", None))

    else:  # cpu
        mdl = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
            local_files_only=True,
        )
        if verbose:
            print(">> device_map:", getattr(mdl, "hf_device_map", None))

    # pad_token fallback / pad_token 예비 설정
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    return tok, mdl


# ========== Build prompt / 프롬프트 ==========
def format_prompt(tok: 'AutoTokenizer', system: str, user: str) -> str:
    """Prefer chat template; otherwise use Llama2 [INST] / 채팅 템플릿 우선, 없으면 [INST]"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"


# ====== New: KB read + retrieval / 신규: 지식베이스 로드 + 검색 ======
def _read_kb(kb_dir: str):
    """Read plain-text files under kb_dir / kb_dir의 텍스트 파일 로드"""
    if not kb_dir:
        return []
    p = Path(kb_dir)
    if not p.exists() or not p.is_dir():
        return []
    docs = []
    for fp in sorted(p.iterdir()):
        if fp.is_file():
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore").strip()
                if txt:
                    docs.append(txt)
            except Exception:
                pass
    return docs


class _TFIDFRetriever:
    """TF-IDF + cosine approximate retrieval (lightweight) / TF-IDF + 코사인 근사 검색(경량)"""
    def __init__(self, docs):
        self.docs = docs
        if docs:
            self.vec = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2))
            self.mat = self.vec.fit_transform(docs)
        else:
            self.vec, self.mat = None, None

    def search(self, query: str, k: int = 5):
        if not self.docs:
            return []
        qv = self.vec.transform([query])
        sims = cosine_similarity(qv, self.mat)[0]
        idx = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in idx]


def _build_context_block(docs, hits):
    """Assemble matched text into [CONTEXT] / 적중 텍스트를 [CONTEXT]로 결합"""
    if not hits:
        return ""
    parts = []
    for i, sc in hits:
        snippet = docs[i][:1200]
        parts.append(f"[Doc#{len(parts)+1} | score={sc:.3f}]\n{snippet}\n")
    return "[CONTEXT]\n" + "\n".join(parts) + "\n"


def _compose_user_with_context(user_query: str, context_block: str):
    """Combine evidence and question into the user content / 증거와 질문을 사용자 프롬프트에 결합"""
    role_rules = (
        "You are a rigorous, fact-oriented assistant. Answer only based on the above “[CONTEXT] known information” "
        "and basic common facts; if evidence is insufficient, say “insufficient evidence” and specify exactly what "
        "additional information is needed. Do not fabricate numbers, dates, proper nouns, or sources."
    )
    # Keep the bilingual comment style / 한중 이중 주석 스타일 유지
    return (
        f"{role_rules}\n"
        f"{context_block}"
        "[QUERY]\n"
        f"{user_query}\n\n"
        "[ANSWER]\n"
    )


# ====== New: consistency/entropy/voting evaluation and decision / 신규: 일치성/엔트로피/투표 평가와 판정 ======
def _answer_consistency(answer: str, evidence_texts):
    """TF-IDF consistency score between answer and evidence (0~1) / 정답-증거 TF-IDF 일치성(0~1)"""
    if not answer.strip() or not evidence_texts:
        return 0.0
    vect = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2))
    mat = vect.fit_transform(evidence_texts + [answer])
    E = mat[:-1]
    A = mat[-1]
    sims = cosine_similarity(A, E)[0]
    return float(sims.max()) if sims.size else 0.0


def _self_consistency_center(candidates):
    """Self-consistency among candidates: pick the one with the highest mean similarity to others / 후보간 자기일치: 평균 유사도 최고"""
    if len(candidates) == 1:
        return candidates[0], 1.0
    vect = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2))
    X = vect.fit_transform(candidates)
    sims = cosine_similarity(X, X).mean(axis=1)
    i = int(sims.argmax())
    return candidates[i], float(sims[i])


def _last_token_entropy(model, inputs):
    """Approximate uncertainty: entropy of the distribution at the last position / 근사 불확실성: 마지막 위치 분포의 엔트로피"""
    try:
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :]
        prob = torch.softmax(logits, dim=-1)
        ent = (-prob * torch.log(prob + 1e-12)).sum(-1)
        return float(ent.mean().item())
    except Exception:
        return 0.0


# ========== Main / 메인 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",
                    default="/home/mew/mev/llm/llama2",
                    help="Local model directory / 로컬 모델 디렉터리")
    ap.add_argument("--mode",
                    choices=["auto","fp16_gpu","int8","fp16_offload","cpu"],
                    default="auto", help="Inference mode / 추론 모드")
    ap.add_argument("--question",
                    default="Please You are a strictly fact-grounded AI assistant for medical/clinical content.",  # please: ...
                    help="User question / 사용자 질문")
    ap.add_argument("--system",
                    default="You are a factual, cautious AI assistant. "
                    "Never fabricate information. "
                    "If the answer is uncertain, explicitly say '정보가 부족합니다' or '정보 부족'. "
                    "Always prioritize verified, logical reasoning over speculation.",
                    help="System prompt / 시스템 프롬프트")
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="Max tokens to generate / 생성 최대 토큰")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature / 샘플링 온도")
    ap.add_argument("--top_p", type=float, default=0.95,
                    help="Nucleus sampling threshold / top-p 값")
    ap.add_argument("--top_k", type=int, default=50,
                    help="Top-k sampling / top-k 샘플링")
    ap.add_argument("--repetition_penalty", type=float, default=1.05,
                    help="Repetition penalty / 반복 페널티")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed / 랜덤 시드")
    ap.add_argument("--stream", action="store_true",
                    help="Streaming output / 스트리밍 출력")
    ap.add_argument("--verbose", action="store_true",
                    help="Show device map and debug info / 디바이스 매핑 등 디버그 정보 표시")

    # ====== New args: retrieval + anti-hallucination thresholds / 신규 파라미터: 검색 + 환각 방지 임계값 ======
    ap.add_argument("--kb_dir", type=str, default="",
                    help="Knowledge base directory (plain txt/md) / 지식베이스 디렉터리")
    ap.add_argument("--evidence_topk", type=int, default=6,
                    help="Number of evidence items to retrieve / 검색 증거 개수")
    ap.add_argument("--n_samples", type=int, default=3,
                    help="Number of diverse samples for self-consistency voting / 다중 샘플 수")
    ap.add_argument("--guard_consistency", type=float, default=0.22,
                    help="Minimum consistency threshold (effective when KB exists) / 일치성 최저 임계치(KB 있을 때)")
    ap.add_argument("--guard_entropy_max", type=float, default=6.2,
                    help="Approximate entropy upper bound / 근사 엔트로피 상한")
    ap.add_argument("--guard_consensus_min", type=float, default=0.26,
                    help="Minimum self-consensus threshold / 자기 일치 최저 임계치")
    ap.add_argument("--guard_enabled", action="store_true",
                    help="Enable anti-hallucination decision (recommended) / 환각 방지 판정 활성화")

    args = ap.parse_args()

    mode = pick_mode(args.mode)
    if args.verbose:
        print(f">> selected mode / 선택된 모드: {mode}")

    tok, mdl = build_model(args.model_dir, mode=mode, verbose=args.verbose)

    # Randomness / 랜덤성
    torch.manual_seed(args.seed)

    # ====== Retrieval stage / 검색 단계 ======
    docs = _read_kb(args.kb_dir)
    retr = _TFIDFRetriever(docs) if docs else None
    hits = retr.search(args.question, k=args.evidence_topk) if retr else []
    context_block = _build_context_block(docs, hits)
    user_with_ctx = _compose_user_with_context(args.question, context_block) if context_block else args.question

    # Prompt / 프롬프트
    prompt = format_prompt(tok, args.system, user_with_ctx)

    # Move ONLY input tensors to GPU; do NOT call model.to(...)
    # 입력 텐서만 GPU로 이동; 모델에는 .to(...) 호출 금지
    inputs = tok(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        # ---- Generation config ----
    do_sample = args.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        do_sample=do_sample,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )

    if do_sample:
        gen_kwargs.update(dict(
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            top_k=args.top_k,
        ))

    # ★ Key fix: allow multiple outputs under greedy by switching to beam; sampling still uses num_return_sequences
    # ★ 关键修复：允许在 greedy 下返回多条，用 beam 来承载；采样下仍用 num_return_sequences
    if args.n_samples > 1:
        if do_sample:
            gen_kwargs["num_return_sequences"] = args.n_samples
        else:
            # Greedy does not support multiple returns -> switch to beam automatically
            # greedy는 다중 반환 불가 -> beam으로 자동 전환
            gen_kwargs["num_beams"] = max(2, args.n_samples)  # at least 2
            gen_kwargs["num_return_sequences"] = args.n_samples
            gen_kwargs.setdefault("early_stopping", True)

    # (Optional) Remove warnings like "do_sample=False but temperature/top_p set"
    # (선택) "do_sample=False인데 temperature/top_p를 설정" 경고 제거
    if not do_sample:
        try:
            mdl.generation_config.temperature = None
            mdl.generation_config.top_p = None
            mdl.generation_config.top_k = None
        except Exception:
            pass

    # ====== Coordination between guard mode and streaming / 판정 모드와 스트리밍의 조정 ======
    # Anti-hallucination needs multi-sample metrics; streaming is unsuitable.
    # If both are enabled, automatically downgrade to non-streaming.
    # 환각 방지는 다중 샘플과 측정이 필요 → 스트리밍과 병행 어려움; 동시 요청 시 비스트리밍으로 강등
    stream_allowed = args.stream and (not args.guard_enabled)
    if args.stream and args.guard_enabled:
        print(">> [Notice] Guard mode enabled; streaming is disabled to allow scoring & voting.\n")

    # ====== Generation: multi-sample candidates / 생성: 다중 후보 ======
    with torch.inference_mode():
        if stream_allowed:
            # —— Streaming output / 스트리밍 출력 ——
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs_stream = dict(gen_kwargs)
            gen_kwargs_stream["streamer"] = streamer

            def _worker():
                with torch.inference_mode():
                    mdl.generate(**inputs, **gen_kwargs_stream)
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            for text in streamer:
                print(text, end="", flush=True)
            print()
            return
        else:
            # Non-streaming: generate n_samples in one shot / 비스트리밍: 일괄 n_samples
            if args.n_samples <= 1:
                gen_ids = mdl.generate(**inputs, **gen_kwargs)
                cand_texts = [tok.decode(gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()]
            else:
                # num_return_sequences returns multiple sequences at once / 한 번에 다중 시퀀스
                gen_ids = mdl.generate(**inputs, **gen_kwargs)
                cand_texts = []
                step = gen_ids.size(0)
                for i in range(step):
                    new_tokens = gen_ids[i, inputs["input_ids"].shape[1]:]
                    cand_texts.append(tok.decode(new_tokens, skip_special_tokens=True).strip())

    # ====== Approximate entropy (distribution at end of prompt) / 근사 엔트로피 추정 ======
    entropy_est = _last_token_entropy(mdl, inputs)

    # ====== Self-consistency voting / 자기 일치 투표 ======
    voted_text, consensus = _self_consistency_center(cand_texts)

    # ====== Consistency score (answer vs evidence) / 일치성 점수(정답 vs 증거) ======
    evidence_texts = [docs[i] for i, _ in hits] if hits else []
    consistency = _answer_consistency(voted_text, evidence_texts) if evidence_texts else 0.0

    # ====== Decision / 판정 ======
    decision = "answer"
    reasons = []
    if args.guard_enabled:
        # Only check consistency threshold when KB exists / KB 있을 때만 일치성 임계 검사
        if evidence_texts and consistency < args.guard_consistency:
            decision = "insufficient_evidence"
            reasons.append(f"consistency {consistency:.3f} < {args.guard_consistency}")
        if entropy_est > args.guard_entropy_max:
            decision = "low_confidence"
            reasons.append(f"entropy {entropy_est:.2f} > {args.guard_entropy_max}")
        if consensus < args.guard_consensus_min:
            decision = "low_consensus"
            reasons.append(f"consensus {consensus:.3f} < {args.guard_consensus_min}")

    # ====== Auto-completion and output cleanup / 자동 보완 및 출력 정리 ======
    answer = voted_text
    # Detect incomplete ending / 문장 끝 완성 여부 체크
    if not answer.endswith(('.', '。', '!', '！', '?', '？', '"', '”')):
        # Lightweight continuation to avoid abrupt truncation / 경량 이어쓰기
        continuation_input = tok(answer, return_tensors="pt")
        if torch.cuda.is_available():
            continuation_input = {k: v.to("cuda:0") for k, v in continuation_input.items()}
        with torch.inference_mode():
            continuation_ids = mdl.generate(
                **continuation_input,
                max_new_tokens=80,
                temperature=max(args.temperature, 0.3),
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
        cont = tok.decode(continuation_ids[0, continuation_input["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if cont:
            answer = (answer + " " + cont).strip()

    # Replace with a safe response after decision / 판정 후 안전 응답으로 대체
    if args.guard_enabled and decision != "answer":
        safe_msg_en = "Insufficient evidence. Please provide a more specific source, timeframe, or authoritative data (e.g., a standard ID, paper, or official manual) so I can give a verifiable answer."
        if evidence_texts:
            safe_msg_en += " (With current evidence, I will remain conservative to avoid inventing numbers or proper nouns.)"
        safe_msg_kr = "정보가 부족합니다. 표준 번호/논문/공식 문서 등 더 구체적 근거를 주시면 검증 가능한 답을 드리겠습니다."
        answer = f"{safe_msg_en}\n{safe_msg_kr}"

    # ====== Structured debug info (for logging) / 구조화 디버그 정보 ======
    debug_pack = {
        "decision": decision,
        "reasons": reasons or ["OK"],
        "consistency": round(consistency, 3),
        "self_consensus": round(consensus, 3),
        "entropy_est": round(entropy_est, 3),
        "n_candidates": len(cand_texts),
        "evidence_hit": [
            {"rank": r+1, "sim": round(sc, 3), "preview": (docs[i][:160].replace("\n"," ") if docs else "")}
            for r, (i, sc) in enumerate(hits[:args.evidence_topk])
        ],
    }

    # ====== Output / 출력 ======
    print("\n" + "="*40)
    print(answer)
    print("="*40)
    print(json.dumps(debug_pack, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
