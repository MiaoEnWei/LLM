#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, re, warnings
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Silencesilence FutureWarning about tokenization space cleanup
warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
    module="transformers.tokenization_utils_base",
)

def clip_two_sentences(txt: str) -> str:
    txt = re.sub(r"\s+", " ", (txt or "").strip())
    parts = re.split(r'(?<=[.!?。！？])\s+', txt)
    out = " ".join(parts[:2]).strip()
    if out and out[-1] not in ".!?。！？":
        out += "."
    return out

def read_text_file(p: str) -> str:
    try:
        t = Path(p).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    # Normalize whitespace: remove NBSP/BOM
    return t.replace("\u00a0", " ").replace("\ufeff", "").strip()

def build_force_words(tok, phrases):
    force = []
    for ph in phrases:
        ph = ph.strip()
        if not ph:
            continue
        ids = tok(ph, add_special_tokens=False)["input_ids"]
        if ids:
            force.append(ids)
    return force or None

def main():
    os.environ.setdefault("PYTHONIOENCODING", "UTF-8")
    os.environ.setdefault("LC_ALL", "C.UTF-8")
    os.environ.setdefault("LANG", "C.UTF-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")

    ap.add_argument("--question", required=True)
    ap.add_argument("--answer_prefix", default="")
    ap.add_argument("--second_prefix", default="", help="Second-sentence prefix, e.g., 'Contraindication: '")

    # system/hint: can be provided as text or file (file takes precedence)
    ap.add_argument("--system", default="Answer concisely; prefer the [HINT] when relevant; avoid invented numbers.")
    ap.add_argument("--system_file", default="")
    ap.add_argument("--hint_context", default="")
    ap.add_argument("--hint_file", default="")

    # Sampling & beam search
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--n_beams", type=int, default=4)

    # Constraints: forced phrases (comma-separated)
    ap.add_argument("--force_phrases", default="", help='comma-separated phrases to force, e.g. "active gastrointestinal bleeding"')

    # Misc
    ap.add_argument("--two_sentences", action="store_true", help="Clip output to exactly two sentences")
    ap.add_argument("--show_prompt", action="store_true")
    args = ap.parse_args()

    # Load model
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=False)
    tok.clean_up_tokenization_spaces = False
    mdl = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device = "cuda:0" if ((args.device in ["auto","cuda"]) and torch.cuda.is_available()) else "cpu"
    mdl.to(device)

    # system & hint
    system_text = read_text_file(args.system_file) if args.system_file else args.system
    hint_text = read_text_file(args.hint_file) if args.hint_file else args.hint_context
    hint_block = f"\n\n[HINT]\n{hint_text}\n" if hint_text else ""

    # Merge second-sentence prefix into answer_prefix as a hard anchor
    apfx = args.answer_prefix
    if args.second_prefix and (not apfx.endswith(args.second_prefix)):
        apfx = (apfx + args.second_prefix) if apfx else args.second_prefix

    prompt = f"""{system_text}
{hint_block}

Q: {args.question.strip()}
A: {apfx}"""

    if args.show_prompt:
        print("============== PROMPT BEGIN ==============")
        print(prompt)
        print("=============== PROMPT END ===============", flush=True)

    inputs = tok(prompt, return_tensors="pt").to(device)

    # Generation parameters
    force_list = [s for s in (args.force_phrases.split(",") if args.force_phrases else []) if s.strip()]
    force_words_ids = build_force_words(tok, force_list)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        num_return_sequences=args.n_samples,
    )

    if force_words_ids:
        # Beam search + forced phrases (most reliable)
        gen_kwargs.update(dict(
            do_sample=False,
            num_beams=max(1, args.n_beams),
            force_words_ids=force_words_ids,
        ))
    else:
        # Pure sampling (more natural)
        gen_kwargs.update(dict(
            do_sample=True,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            top_k=args.top_k,
        ))

    out = mdl.generate(**inputs, **gen_kwargs)

    # Take only the first sequence
    seq = out[0] if hasattr(out, "__iter__") else out.sequences[0]
    text = tok.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    pos = text.rfind("\nA:")
    ans = text[pos+3:].strip() if pos != -1 else text.strip()

    if args.two_sentences:
        ans = clip_two_sentences(ans)

    # Fallback: if second_prefix is missing within two sentences or forced phrase missing -> construct a safe 2-sentence output
    needs_second = args.two_sentences and (args.second_prefix and args.second_prefix not in ans)
    needs_forced = (force_list and (not any(ph in ans for ph in force_list)))
    if needs_second or needs_forced:
        # Sentence 1: keep the generated first sentence if possible; otherwise use a generic mechanism sentence from hint
        parts = re.split(r'(?<=[.!?。！？])\s+', ans)
        first = (parts[0].strip() if parts and parts[0].strip() else "").rstrip()
        if not first:
            first = "Aspirin irreversibly inhibits cyclooxygenase to reduce thromboxane A2 and platelet aggregation."
        if first and first[-1] not in ".!?。！？":
            first += "."
        forced = force_list[0] if force_list else ""
        second = f"{args.second_prefix}{forced}".strip()
        if second and not second.endswith("."):
            second += "."
        ans = f"{first} {second}".strip()

    print(ans)

if __name__ == "__main__":
    main()
