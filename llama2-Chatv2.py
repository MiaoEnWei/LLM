#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Llama-2-7B-Chat 本地推理 · 强化版（12GB 显存友好）
Llama-2-7B-Chat 로컬 추론 · 강화판 (12GB VRAM 친화)

- 模式：auto / fp16_gpu / int8 / fp16_offload / cpu
  모드: auto / fp16_gpu / int8 / fp16_offload / cpu
- 仅移动“输入张量”到 GPU；绝不对模型整体 .to(...)（避免 4/8bit 报错）
  입력 텐서만 GPU로 이동, 모델 전체 .to(...) 금지
- int8 分支使用“模块级 device_map”，必要时 lm_head→CPU，避开 bnb 的 .to 冲突
  int8 분기: 모듈 단위 device_map, 필요 시 lm_head를 CPU로 배치
- 支持流式输出(--stream)、低温度保守采样、JSON 友好
  스트리밍 출력, 저온도 보수 샘플링 지원

# ====== 新增：轻量防幻觉机制 / 신규: 경량 환각 방지 메커니즘 ======
# - --kb_dir 指向本地知识库目录（txt/md 纯文本）
# - 检索 Top-K 证据拼入 [CONTEXT]，引导模型“先取证再作答”
# - 生成多样本，自一致性投票 + 一致性评分 + 近似熵 → 裁决（answer / refuse）
# - 无需微调、无需联网，完全黑盒外控
"""

import os, argparse, threading, gc, json, math
from pathlib import Path
import psutil, torch

# ↓ 减少显存碎片（需在 import torch 前设置）/ GPU 메모리 파편화 완화
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig,
    TextIteratorStreamer
)
from transformers.utils import logging as hf_logging
import logging

# ====== 新增依赖：TF-IDF 检索 / 신규 의존성: TF-IDF 검색 ======
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ↓ 默认安静，可用 --verbose 打开 / 기본은 조용, --verbose로 상세 로그
hf_logging.set_verbosity_error()
logging.getLogger("accelerate").setLevel(logging.ERROR)


# ========== 模式选择 / 모드 선택 ==========
def pick_mode(requested: str) -> str:
    """根据请求与硬件选择模式 / 요청 + 하드웨어로 모드 선택"""
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 16:
        return "fp16_gpu"
    elif vram_gb >= 10:
        return "int8"           # 10~16GB → int8 最稳 / 가장 안정
    else:
        return "fp16_offload"   # 更小显存 → FP16+오프로딩


def _cleanup_cuda():
    """释放当前进程持有的显存 / 현재 프로세스의 VRAM 해제"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


# ========== 构建模型 / 모델 로드 ==========
def build_model(model_dir: str, mode: str, verbose: bool=False):
    model_dir = str(Path(model_dir).expanduser().resolve())
    assert Path(model_dir).is_dir(), f"模型目录不存在：{model_dir} / 모델 디렉터리가 없습니다."

    _cleanup_cuda()  # 避免上次失败残留显存 / 이전 실패 잔여 메모리 정리

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    attn_impl = "sdpa"

    # 读取层数 / 레이어 수
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    n_layers = getattr(cfg, "num_hidden_layers", 32)

    if mode == "int8":
        # 8bit 量化（模块级多设备映射）
        # 8bit 양자화(모듈 단위 다중 디바이스 매핑)
        bnb = BitsAndBytesConfig(load_in_8bit=True)

        # 大部分模块放 GPU0，lm_head 放 CPU（小且安全），避免被整体 .to(...)
        # 대부분 GPU0, lm_head는 CPU로 배치하여 전체 .to(...) 회피
        device_map = {"model.embed_tokens": 0, "model.norm": 0, "lm_head": "cpu"}
        device_map.update({f"model.layers.{i}": 0 for i in range(n_layers)})

        offload_dir = os.path.expanduser("~/hf_offload_int8")
        os.makedirs(offload_dir, exist_ok=True)

        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                model_dir,
                quantization_config=bnb,
                device_map=device_map,      # 关键：模块级 device_map / 핵심
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
                local_files_only=True,
                offload_folder=offload_dir,
            )
            if verbose:
                print(">> device_map:", getattr(mdl, "hf_device_map", None))
        except Exception as e:
            # 若仍报错，回退 FP16+offload（更保守）
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
            device_map={"": 0},   # 全 GPU0 / 전부 GPU0
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
            local_files_only=True,
        )
        if verbose:
            print(">> device_map:", getattr(mdl, "hf_device_map", None))

    elif mode == "fp16_offload":
        ram_gb = psutil.virtual_memory().total // (1024 ** 3)
        max_mem = {0: "9.5GiB", "cpu": f"{int(ram_gb * 0.8)}GiB"}  # 更保守 / 보수적
        mdl = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",              # 放不下自动溢出到 CPU / 자동 오프로딩
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

    # pad_token 兜底 / pad_token 예비 설정
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    return tok, mdl


# ========== 构建 Prompt / 프롬프트 ==========
def format_prompt(tok: 'AutoTokenizer', system: str, user: str) -> str:
    """优先 chat 模板；无则 Llama2 [INST] / 채팅 템플릿 우선, 없으면 [INST]"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"


# ====== 新增：知识库读取 + 检索 / 신규: 지식베이스 로드 + 검색 ======
def _read_kb(kb_dir: str):
    """读取 kb_dir 下纯文本文件 / kb_dir의 텍스트 파일 로드"""
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
    """TF-IDF + 余弦近似检索（轻量） / TF-IDF + 코사인 근사 검색(경량)"""
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
    """把命中文本拼成 [CONTEXT] / 적중 텍스트를 [CONTEXT]로 결합"""
    if not hits:
        return ""
    parts = []
    for i, sc in hits:
        snippet = docs[i][:1200]
        parts.append(f"[Doc#{len(parts)+1} | score={sc:.3f}]\n{snippet}\n")
    return "[CONTEXT]\n" + "\n".join(parts) + "\n"


def _compose_user_with_context(user_query: str, context_block: str):
    """把证据与问题拼进用户内容 / 증거와 질문을 사용자 프롬프트에 결합"""
    role_rules = (
        "你是一名严谨的事实型助手。只依据上面的“[CONTEXT] 已知信息”和常识基础事实回答；"
        "若证据不足，请直接说“证据不足”，并指出需要补充的具体信息。禁止编造数据、日期、专名或来源。"
    )
    # 中韩双语风格延续 / 한중 이중 주석 스타일 유지
    return (
        f"{role_rules}\n"
        f"{context_block}"
        "[QUERY]\n"
        f"{user_query}\n\n"
        "[ANSWER]\n"
    )


# ====== 新增：一致性/熵/投票 评估与裁决 / 신규: 일치성/엔트로피/투표 평가와 판정 ======
def _answer_consistency(answer: str, evidence_texts):
    """答案与证据的 TF-IDF 一致性评分（0~1）/ 정답-증거 TF-IDF 일치성(0~1)"""
    if not answer.strip() or not evidence_texts:
        return 0.0
    vect = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2))
    mat = vect.fit_transform(evidence_texts + [answer])
    E = mat[:-1]
    A = mat[-1]
    sims = cosine_similarity(A, E)[0]
    return float(sims.max()) if sims.size else 0.0


def _self_consistency_center(candidates):
    """候选间自一致性：取与他者平均相似度最高者 / 후보간 자기일치: 평균 유사도 최고"""
    if len(candidates) == 1:
        return candidates[0], 1.0
    vect = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2))
    X = vect.fit_transform(candidates)
    sims = cosine_similarity(X, X).mean(axis=1)
    i = int(sims.argmax())
    return candidates[i], float(sims[i])


def _last_token_entropy(model, inputs):
    """近似不确定性：最后位置分布熵 / 근사 불확실성: 마지막 위치 분포의 엔트로피"""
    try:
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :]
        prob = torch.softmax(logits, dim=-1)
        ent = (-prob * torch.log(prob + 1e-12)).sum(-1)
        return float(ent.mean().item())
    except Exception:
        return 0.0


# ========== 主函数 / 메인 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",
                    default="/home/mew/mev/llm/llama2",
                    help="本地模型目录 / 로컬 모델 디렉터리")
    ap.add_argument("--mode",
                    choices=["auto","fp16_gpu","int8","fp16_offload","cpu"],
                    default="auto", help="推理模式 / 추론 모드")
    ap.add_argument("--question",
                    default="请You are a strictly fact-grounded AI assistant for medical/clinical content.",#请以事实为基础，总结重入漏洞的本质和一种通用防护策略，不要使用假设，가설을 사용하지 말고, 사실을 기초로 하여, 다시 들어가는 허점의 본질과 일반적인 방호 전략을 총결산하십시오.
                    help="用户问题 / 사용자 질문")
    ap.add_argument("--system",
                    default="You are a factual, cautious AI assistant. "
                    "Never fabricate information. "
                    "If the answer is uncertain, explicitly say '정보가 부족합니다' or '정보 부족'. "
                    "Always prioritize verified, logical reasoning over speculation.",
                    help="系统提示 / 시스템 프롬프트")
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="生成上限 / 생성 최대 토큰")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="采样温度 / 샘플링 온도")
    ap.add_argument("--top_p", type=float, default=0.95,
                    help="核采样阈值 / top-p 값")
    ap.add_argument("--top_k", type=int, default=50,
                    help="top-k 采样 / top-k 샘플링")
    ap.add_argument("--repetition_penalty", type=float, default=1.05,
                    help="重复惩罚 / 반복 페널티")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子 / 랜덤 시드")
    ap.add_argument("--stream", action="store_true",
                    help="流式输出 / 스트리밍 출력")
    ap.add_argument("--verbose", action="store_true",
                    help="显示设备映射等调试信息 / 디바이스 매핑 등 디버그 정보 표시")

    # ====== 新增参数：检索与防幻觉阈值 / 신규 파라미터: 검색 + 환각 방지 임계값 ======
    ap.add_argument("--kb_dir", type=str, default="",
                    help="知识库目录（txt/md 纯文本）/ 지식베이스 디렉터리")
    ap.add_argument("--evidence_topk", type=int, default=6,
                    help="检索证据条数 / 검색 증거 개수")
    ap.add_argument("--n_samples", type=int, default=3,
                    help="多样本自一致性投票返回数 / 다중 샘플 수")
    ap.add_argument("--guard_consistency", type=float, default=0.22,
                    help="一致性最低阈值（有 KB 时生效）/ 일치성 최저 임계치(KB 있을 때)")
    ap.add_argument("--guard_entropy_max", type=float, default=6.2,
                    help="近似熵上限 / 근사 엔트로피 상한")
    ap.add_argument("--guard_consensus_min", type=float, default=0.26,
                    help="自一致性最低阈值 / 자기 일치 최저 임계치")
    ap.add_argument("--guard_enabled", action="store_true",
                    help="启用防幻觉裁决（推荐开启）/ 환각 방지 판정 활성화")

    args = ap.parse_args()

    mode = pick_mode(args.mode)
    if args.verbose:
        print(f">> selected mode / 선택된 모드: {mode}")

    tok, mdl = build_model(args.model_dir, mode=mode, verbose=args.verbose)

    # 随机性 / 랜덤성
    torch.manual_seed(args.seed)

    # ====== 检索阶段 / 검색 단계 ======
    docs = _read_kb(args.kb_dir)
    retr = _TFIDFRetriever(docs) if docs else None
    hits = retr.search(args.question, k=args.evidence_topk) if retr else []
    context_block = _build_context_block(docs, hits)
    user_with_ctx = _compose_user_with_context(args.question, context_block) if context_block else args.question

    # prompt / 프롬프트
    prompt = format_prompt(tok, args.system, user_with_ctx)

    # 仅把“输入张量”放到 GPU；不要对模型 .to(...)
    # 입력 텐서만 GPU로 이동; 모델에는 .to(...) 호출 금지
    inputs = tok(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        # ---- 生成参数 / Generation config ----
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

    # ★ 关键修复：允许在 greedy 下返回多条，用 beam 来承载；采样下仍用 num_return_sequences
    if args.n_samples > 1:
        if do_sample:
            gen_kwargs["num_return_sequences"] = args.n_samples
        else:
            # greedy 不支持多返回，自动切换到 beam
            gen_kwargs["num_beams"] = max(2, args.n_samples)  # 至少 2
            gen_kwargs["num_return_sequences"] = args.n_samples
            gen_kwargs.setdefault("early_stopping", True)

    #（可选）消除“do_sample=False 但设了 temperature/top_p 的警告”
    if not do_sample:
        try:
            mdl.generation_config.temperature = None
            mdl.generation_config.top_p = None
            mdl.generation_config.top_k = None
        except Exception:
            pass

    # ====== 裁决模式与流式输出的协调 / 판정 모드와 스트리밍의 조정 ======
    # 防幻觉需要多样本与度量，流式输出不适合；若二者同时启用，则自动降级为非流式。
    # 환각 방지는 다중 샘플과 측정이 필요 → 스트리밍과 병행 어려움; 동시 요청 시 비스트리밍으로 강등
    stream_allowed = args.stream and (not args.guard_enabled)
    if args.stream and args.guard_enabled:
        print(">> [Notice] Guard mode enabled; streaming is disabled to allow scoring & voting.\n")

    # ====== 生成：多样本候选 / 생성: 다중 후보 ======
    with torch.inference_mode():
        if stream_allowed:
            # —— 流式输出 / 스트리밍 출력 ——
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
            # 非流式：一次性生成 n_samples / 비스트리밍: 일괄 n_samples
            if args.n_samples <= 1:
                gen_ids = mdl.generate(**inputs, **gen_kwargs)
                cand_texts = [tok.decode(gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()]
            else:
                # num_return_sequences 一次返回多条 / 한 번에 다중 시퀀스
                gen_ids = mdl.generate(**inputs, **gen_kwargs)
                cand_texts = []
                step = gen_ids.size(0)
                for i in range(step):
                    new_tokens = gen_ids[i, inputs["input_ids"].shape[1]:]
                    cand_texts.append(tok.decode(new_tokens, skip_special_tokens=True).strip())

    # ====== 近似熵估计（用提示末位的分布）/ 근사 엔트로피 추정 ======
    entropy_est = _last_token_entropy(mdl, inputs)

    # ====== 自一致性投票 / 자기 일치 투표 ======
    voted_text, consensus = _self_consistency_center(cand_texts)

    # ====== 一致性评分（答案 vs 证据）/ 일치성 점수(정답 vs 증거) ======
    evidence_texts = [docs[i] for i, _ in hits] if hits else []
    consistency = _answer_consistency(voted_text, evidence_texts) if evidence_texts else 0.0

    # ====== 裁决 / 판정 ======
    decision = "answer"
    reasons = []
    if args.guard_enabled:
        # 有 KB 时才检查一致性阈值 / KB 있을 때만 일치성 임계 검사
        if evidence_texts and consistency < args.guard_consistency:
            decision = "insufficient_evidence"
            reasons.append(f"consistency {consistency:.3f} < {args.guard_consistency}")
        if entropy_est > args.guard_entropy_max:
            decision = "low_confidence"
            reasons.append(f"entropy {entropy_est:.2f} > {args.guard_entropy_max}")
        if consensus < args.guard_consensus_min:
            decision = "low_consensus"
            reasons.append(f"consensus {consensus:.3f} < {args.guard_consensus_min}")

    # ====== 自动补完与输出整理 / 자동 보완 및 출력 정리 ======
    answer = voted_text
    # 自动检测句尾是否完整
    if not answer.endswith(('.', '。', '!', '！', '?', '？', '"', '”')):
        # 轻量续写，避免突兀 / 경량 이어쓰기
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

    # 裁决后替换为安全回应 / 판정 후 안전 응답으로 대체
    if args.guard_enabled and decision != "answer":
        safe_msg_cn = "证据不足。请提供更具体的来源、时间或权威数据（如标准编号/论文/官方手册），我再给出可核查答案。"
        if evidence_texts:
            safe_msg_cn += "（基于当前证据只能给出保守描述，避免编造具体数值或专名。）"
        safe_msg_kr = "정보가 부족합니다. 표준 번호/논문/공식 문서 등 더 구체적 근거를 주시면 검증 가능한 답을 드리겠습니다."
        answer = f"{safe_msg_cn}\n{safe_msg_kr}"

    # ====== 结构化调试信息（便于日志）/ 구조화 디버그 정보 ======
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

    # ====== 输出 / 출력 ======
    print("\n" + "="*40)
    print(answer)
    print("="*40)
    print(json.dumps(debug_pack, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
