# File name: scripts/eval_compare_models_korean.py
# New script: Compare MedMCQA performance between base GPT-2 and the prompt tuning model (English output)
# - Run both models at the same time
# - Categorize: both correct, both wrong, base correct & prompt tuning wrong, base wrong & prompt tuning correct
# - Randomly print 5 examples per category (or all if fewer): index, prompt, gold, pred_base, pred_prompt_tuning

import argparse, json, re, os
import random
import numpy as np
from typing import Optional, Tuple, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LETTERS = "ABCD"

# ---- PEFT support ----
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="./gpt2",
        help="Base model path (e.g., ./gpt2)",
    )
    ap.add_argument(
        "--adapter",
        default=".\out_gpt2_pubmedqa_decision_prompt_tuning_v2",
        help="PEFT adapter directory (e.g., .\out_gpt2_pubmedqa_decision_prompt_tuning_v2)",
    )
    ap.add_argument(
        "--val",
        default=".\data\official_raw\medmcqa_validation.jsonl",
        help="Validation data jsonl path",
    )
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    ap.add_argument(
        "--dtype",
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    ap.add_argument(
        "--calib_n",
        type=int,
        default=1500,
        help="Number of pre-samples; 0=disable calibration",
    )
    ap.add_argument(
        "--device_map",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    ap.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="bitsandbytes 8-bit quantization",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, evaluate only the first N samples (for debugging)",
    )
    ap.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples to print per category",
    )
    return ap.parse_args()


def str2dtype(x: str):
    return {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[x]


# parsing functions (same as original)
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
    tail = s[colon + 1 :].strip()
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
    q, a, b, c, d = (
        o.get("question"),
        o.get("opa"),
        o.get("opb"),
        o.get("opc"),
        o.get("opd"),
    )
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
    return prompts, gold


# scoring functions (same as original)
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
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    enc = {k: v.to(dev) for k, v in enc.items()}
    logits = model(**enc).logits[:, -1, :]
    return torch.log_softmax(logits, dim=-1).to(torch.float32)


@torch.inference_mode()
def estimate_prior(model, tok, texts, max_len, cand, batch, dev):
    s = {ch: [] for ch in LETTERS}
    for i in range(0, len(texts), batch):
        lp = last_logprobs(model, tok, texts[i : i + batch], max_len, dev)
        for ch in LETTERS:
            idx = cand[ch]
            if not idx:
                s[ch].append(torch.full((lp.size(0),), -1e9, device=lp.device))
            else:
                s[ch].append(
                    torch.logsumexp(torch.stack([lp[:, j] for j in idx], dim=1), dim=1)
                )
    pri = {}
    for ch in LETTERS:
        pri[ch] = torch.cat(s[ch], dim=0).mean().item()
    return pri


def predict(model, tok, prompts, max_len, batch, calib_n, dev, CAND):
    prior = None
    if calib_n and calib_n > 0:
        n = min(calib_n, len(prompts))
        prior = estimate_prior(
            model, tok, prompts[:n], max_len, CAND, batch, dev
        )

    preds = []
    for i in range(0, len(prompts), batch):
        lp = last_logprobs(model, tok, prompts[i : i + batch], max_len, dev)
        cols = []
        for ch in LETTERS:
            idx = CAND[ch]
            if not idx:
                cols.append(torch.full((lp.size(0),), -1e9, device=lp.device))
            else:
                sc = torch.logsumexp(
                    torch.stack([lp[:, j] for j in idx], dim=1), dim=1
                )
                if prior is not None:
                    sc = sc - prior[ch]
                cols.append(sc)
        S = torch.stack(cols, dim=1)
        idx = S.argmax(dim=1).tolist()
        preds.extend(LETTERS[k] for k in idx)
    return preds


def pick_first_cuda_device(model):
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, int) or (isinstance(v, str) and v.startswith("cuda")):
                return torch.device(f"cuda:{v}" if isinstance(v, int) else v)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args, adapter_path=None):
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = str2dtype(args.dtype)
    loader_kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True)

    if args.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig

            loader_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )
        except Exception:
            loader_kwargs["load_in_8bit"] = True

    if args.device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", **loader_kwargs
        ).eval()
        dev = pick_first_cuda_device(model)
    elif args.device_map == "cpu":
        model = (
            AutoModelForCausalLM.from_pretrained(args.model, **loader_kwargs)
            .eval()
            .to("cpu")
        )
        dev = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = (
            AutoModelForCausalLM.from_pretrained(args.model, **loader_kwargs)
            .eval()
            .to(device)
        )
        dev = device

    if adapter_path:
        if not PEFT_AVAILABLE:
            raise ImportError("peft library required: pip install peft")
        print(f"[compare] Loading PEFT adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path).eval()

    model.config.pad_token_id = tok.eos_token_id
    model.config.use_cache = False

    return model, tok, dev


def main():
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = build_args()
    set_seed(args.seed)

    prompts, gold = load_eval_items(args.val)
    assert len(prompts) > 0, f"Parsed 0 samples: {args.val}"

    if args.limit and args.limit > 0:
        n = min(args.limit, len(prompts))
        prompts = prompts[:n]
        gold = gold[:n]
        print(f"[Eval] Limited to the first {n} validation samples from {args.val}")

    # load base model
    print("[compare] Loading base model...")
    model_base, tok, dev = load_model(args)
    CAND = build_candidates(tok)
    assert any(len(v) > 0 for v in CAND.values()), f"No candidate tokens: {CAND}"
    preds_base = predict(model_base, tok, prompts, args.max_len, args.batch, args.calib_n, dev, CAND)

    # load prompt tuning model
    print("[compare] Loading prompt tuning model...")
    model_pt, _, _ = load_model(args, args.adapter)
    preds_pt = predict(model_pt, tok, prompts, args.max_len, args.batch, args.calib_n, dev, CAND)

    # categorize examples
    both_correct = []                     # both correct
    both_wrong = []                       # both wrong
    base_correct_pt_wrong = []            # base correct, prompt tuning wrong
    base_wrong_pt_correct = []            # base wrong, prompt tuning correct

    for i in range(len(prompts)):
        p_base = preds_base[i]
        p_pt = preds_pt[i]
        g = gold[i]
        item = (i, prompts[i], g, p_base, p_pt)

        if p_base == g and p_pt == g:
            both_correct.append(item)
        elif p_base != g and p_pt != g:
            both_wrong.append(item)
        elif p_base == g and p_pt != g:
            base_correct_pt_wrong.append(item)
        elif p_base != g and p_pt == g:
            base_wrong_pt_correct.append(item)

    # random sampling
    def sample_examples(lst, n):
        if len(lst) <= n:
            return lst
        return random.sample(lst, n)

    num = args.num_examples
    both_correct_samples = sample_examples(both_correct, num)
    both_wrong_samples = sample_examples(both_wrong, num)
    base_correct_pt_wrong_samples = sample_examples(base_correct_pt_wrong, num)
    base_wrong_pt_correct_samples = sample_examples(base_wrong_pt_correct, num)

    # English printing function
    def print_samples(title, samples):
        print("\n" + "=" * 20 + f" {title} ({len(samples)} items) " + "=" * 20)
        for (i, prompt, g, p_base, p_pt) in samples:
            print(f"\n--- [Question #{i}] Base: {p_base} | Prompt Tuning: {p_pt} | Gold: {g} ---")
            print(prompt)

    print_samples("① Both Correct", both_correct_samples)
    print_samples("② Both Wrong", both_wrong_samples)
    print_samples("③ Only Base Correct → Prompt Tuning Wrong", base_correct_pt_wrong_samples)
    print_samples("④ Base Wrong → Prompt Tuning Correct (Rescued!)", base_wrong_pt_correct_samples)

    # additional stats (version without the previous bug)
    total = len(prompts)
    acc_base = sum(1 for i in range(total) if preds_base[i] == gold[i]) / total
    acc_pt = sum(1 for i in range(total) if preds_pt[i] == gold[i]) / total

    print("\n" + "="*60)
    print(f"Total questions         : {total} items")
    print(f"Base model accuracy      : {acc_base*100:5.2f}%")
    print(f"Prompt Tuning accuracy   : {acc_pt*100:5.2f}%  (+{ (acc_pt - acc_base)*100 :5.2f}%)")
    print(f"Questions rescued by Prompt Tuning : {len(base_wrong_pt_correct)} items")
    print("="*60)


if __name__ == "__main__":
    main()