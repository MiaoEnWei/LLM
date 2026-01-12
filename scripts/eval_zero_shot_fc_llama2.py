# scripts/eval_zero_shot_fc.py
# Zero-shot 4-choice eval (final-token log-prob) with calibration, detailed metrics,
# and safe loading for LLaMA-2 via device_map=auto / 8-bit.
import argparse, json, re, os
from typing import Optional, Tuple, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LETTERS = "ABCD"

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/mew/mev/llm/llama2")
    ap.add_argument("--val", default="data/official_instruct/medmcqa_validation.jsonl")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=24)  # LLaMA-2: 16~32 is usually stable
    ap.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--calib_n", type=int, default=1500, help="Number of prior samples; 0=disable calibration")
    # --- Key: add these two arguments ---
    ap.add_argument("--device_map", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto=layerwise CPU/GPU offload (recommended)")
    ap.add_argument("--load_in_8bit", action="store_true", help="bitsandbytes 8-bit quantization to save VRAM")
    return ap.parse_args()

def str2dtype(x: str):
    return {"auto": None, "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[x]

# ---------- parsing ----------
def cop_to_letter(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s.upper() in LETTERS:
        return s.upper()
    if s.isdigit():
        k = int(s)
        if 1 <= k <= 4:
            return LETTERS[k - 1]
        if 0 <= k <= 3:
            return LETTERS[k]
    return None

def normalize_answer_prompt(prompt: str) -> str:
    tail = "Answer (A, B, C, or D): "
    if not isinstance(prompt, str):
        return tail
    s = prompt.rstrip("\n")
    low = s.lower()
    i = low.rfind("answer")
    if i >= 0:
        return s[:i] + tail
    if not s.endswith("\n"):
        s += "\n"
    return s + tail

def parse_text_field(o: Dict) -> Optional[Tuple[str, Optional[str]]]:
    t = o.get("text")
    if not isinstance(t, str):
        return None
    s = t.rstrip()
    low = s.lower()
    i = low.rfind("answer")
    if i < 0:
        return None
    m = re.search(r":", s[i:])
    if not m:
        return None
    colon = i + m.start()
    tail = s[colon + 1:].strip()
    gold = next((ch for ch in tail if ch in LETTERS), None)
    prompt = normalize_answer_prompt(s[:i])
    return prompt, gold

def parse_prompt_answer(o: Dict) -> Optional[Tuple[str, Optional[str]]]:
    p = o.get("prompt")
    if not isinstance(p, str):
        return None
    gold = None
    for k in ["answer", "label", "gold", "target", "solution", "correct"]:
        if k in o:
            gold = cop_to_letter(o.get(k))
            break
    prompt = normalize_answer_prompt(p)
    return prompt, gold

def parse_medmcqa(o: Dict) -> Optional[Tuple[str, Optional[str]]]:
    q, a, b, c, d = o.get("question"), o.get("opa"), o.get("opb"), o.get("opc"), o.get("opd")
    if not all(isinstance(x, str) for x in [q, a, b, c, d]):
        return None
    cop = cop_to_letter(o.get("cop"))
    body = (
        "You are a medical exam solver. Choose the single best option and reply with only one letter.\n"
        f"Question: {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n"
    )
    prompt = normalize_answer_prompt(body)
    return prompt, cop

def load_eval_items(path: str) -> Tuple[List[str], List[str]]:
    prompts, gold = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            for fn in (parse_text_field, parse_prompt_answer, parse_medmcqa):
                r = fn(o)
                if r is not None:
                    p, g = r
                    if g is not None:
                        prompts.append(p)
                        gold.append(g)
                    break
    print(prompts)
    return prompts, gold

# ---------- scoring ----------
def single_token_id(tok, s: str) -> Optional[int]:
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None

def build_candidates(tok) -> Dict[str, List[int]]:
    cand = {ch: [single_token_id(tok, ch), single_token_id(tok, " " + ch)] for ch in LETTERS}
    for ch in LETTERS:
        cand[ch] = [i for i in cand[ch] if i is not None]
    return cand

@torch.inference_mode()
def last_logprobs(model, tok, texts: List[str], max_len: int, dev: torch.device):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    enc = {k: v.to(dev) for k, v in enc.items()}
    logits = model(**enc).logits[:, -1, :]
    return torch.log_softmax(logits, dim=-1).to(torch.float32)

@torch.inference_mode()
def estimate_prior(model, tok, texts, max_len, cand, batch, dev):
    s = {ch: [] for ch in LETTERS}
    for i in range(0, len(texts), batch):
        lp = last_logprobs(model, tok, texts[i:i + batch], max_len, dev)
        for ch in LETTERS:
            idx = cand[ch]
            if not idx:
                s[ch].append(torch.full((lp.size(0),), -1e9, device=lp.device))
            else:
                s[ch].append(torch.logsumexp(torch.stack([lp[:, j] for j in idx], dim=1), dim=1))
    pri = {}
    for ch in LETTERS:
        pri[ch] = torch.cat(s[ch], dim=0).mean().item()
    return pri

# ---------- metrics ----------
def compute_confusion_and_metrics(preds: List[str], gold: List[str]) -> None:
    labels = list(LETTERS)
    L = len(labels)
    idx = {c: i for i, c in enumerate(labels)}
    conf = [[0] * L for _ in range(L)]
    for p, g in zip(preds, gold):
        conf[idx[g]][idx[p]] += 1
    pred_cnt = {c: 0 for c in labels}
    gold_cnt = {c: 0 for c in labels}
    for i, c in enumerate(labels):
        gold_cnt[c] = sum(conf[i])
        pred_cnt[c] = sum(conf[r][i] for r in range(L))

    def safe(a, b):
        return (a / b) if b > 0 else 0.0

    by_cls = {}
    for i, c in enumerate(labels):
        TP = conf[i][i]
        FP = pred_cnt[c] - TP
        FN = gold_cnt[c] - TP
        prec = safe(TP, TP + FP)
        rec = safe(TP, TP + FN)
        f1 = safe(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
        by_cls[c] = dict(precision=prec, recall=rec, f1=f1, support=gold_cnt[c], correct=TP)

    total = len(gold)
    acc = sum(conf[i][i] for i in range(L)) / total if total else 0.0
    macro_p = sum(by_cls[c]["precision"] for c in labels) / L
    macro_r = sum(by_cls[c]["recall"] for c in labels) / L
    macro_f = sum(by_cls[c]["f1"] for c in labels) / L
    weighted_f = sum(by_cls[c]["f1"] * by_cls[c]["support"] for c in labels) / total if total else 0.0

    print("\n=== Distribution ===")
    print("Pred count:", pred_cnt)
    print("Gold count:", gold_cnt)
    print("\n=== Per-class ===")
    print("Class  |  Prec   Recall   F1     Support   Correct/Gold")
    for c in labels:
        d = by_cls[c]
        print(f"  {c}    |  {d['precision']:.4f}  {d['recall']:.4f}  {d['f1']:.4f}    {d['support']:5d}     {d['correct']}/{d['support']}")
    print("\n=== Averages ===")
    print(f"Macro-Avg  P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f:.4f}")
    print(f"Weighted   F1={weighted_f:.4f}")
    print(f"Micro-Avg  P={acc:.4f}  R={acc:.4f}  F1={acc:.4f}")
    print(f"ACC        = {acc:.4f}")
    print("\n=== Confusion Matrix (rows=Gold, cols=Pred) ===")
    print("      " + "  ".join(labels))
    for i, c in enumerate(labels):
        row = "  ".join(f"{conf[i][j]:5d}" for j in range(L))
        print(f"{c} | {row}")

def pick_first_cuda_device(model):
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, int) or (isinstance(v, str) and v.startswith("cuda")):
                return torch.device(f"cuda:{v}" if isinstance(v, int) else v)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    args = build_args()
    prompts, gold = load_eval_items(args.val)
    assert len(prompts) > 0, f"Parsed 0 samples: {args.val}"

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # --- LLaMA-2-friendly loading ---
    dtype = str2dtype(args.dtype)
    loader_kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if args.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            loader_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            loader_kwargs["load_in_8bit"] = True  # Backward compatibility for older versions

    if args.device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", **loader_kwargs).eval()
        dev = pick_first_cuda_device(model)
    elif args.device_map == "cpu":
        model = AutoModelForCausalLM.from_pretrained(args.model, **loader_kwargs).eval().to("cpu")
        dev = torch.device("cpu")
    else:  # "cuda"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(args.model, **loader_kwargs).eval().to(device)
        dev = device

    model.config.pad_token_id = tok.eos_token_id
    model.config.use_cache = False

    CAND = build_candidates(tok)
    assert any(len(v) > 0 for v in CAND.values()), f"Empty candidate tokens: {CAND}"

    prior = None
    if args.calib_n and args.calib_n > 0:
        n = min(args.calib_n, len(prompts))
        prior = estimate_prior(model, tok, prompts[:n], args.max_len, CAND, args.batch, dev)

    preds = []
    for i in range(0, len(prompts), args.batch):
        lp = last_logprobs(model, tok, prompts[i:i + args.batch], args.max_len, dev)
        cols = []
        for ch in LETTERS:
            idx = CAND[ch]
            if not idx:
                cols.append(torch.full((lp.size(0),), -1e9, device=lp.device))
            else:
                sc = torch.logsumexp(torch.stack([lp[:, j] for j in idx], dim=1), dim=1)
                if prior is not None:
                    sc = sc - prior[ch]
                cols.append(sc)
        S = torch.stack(cols, dim=1)
        idx = S.argmax(dim=1).tolist()
        preds.extend(LETTERS[k] for k in idx)

    compute_confusion_and_metrics(preds, gold)

if __name__ == "__main__":
    main()
