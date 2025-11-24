#!/usr/bin/env python3
import os, json, shlex, subprocess, sys, re

PROMPT_FILE = "/home/mew/mev/llm/prompts/prompts_cn.txt"
PROPOSER_OUT = "/home/mew/mev/llm/output/proposer_result.json"
FINAL_OUT = "/home/mew/mev/llm/output/final_verifier_result.json"
LLAMA_SCRIPT = "/home/mew/mev/llm/llama2-Chatv2.py"
MODEL_DIR = "/home/mew/mev/llm/llama2"
SOURCE_SOL = "/home/mew/mev/llm/sodility/Reentrancy.sol"

def read_file(p):
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()

def extract_json_like(text):
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        m = re.search(r'(\{(?:[^{}]|\{[^{}]*\})*\})', text, flags=re.S)
        if m:
            return m.group(1)
    return text.strip()

def render_verifier_prompt(function_code, draft_json):
    draft_escaped = draft_json.replace('"', '\\"')
    env = os.environ.copy()
    env["FUNCTION_CODE"] = function_code
    env["DRAFT_JSON"] = draft_escaped
    cmd = f"envsubst '$FUNCTION_CODE $DRAFT_JSON' < {shlex.quote(PROMPT_FILE)} | sed -n '/# VERIFIER/,$p'"
    out = subprocess.check_output(cmd, shell=True, env=env, text=True)
    return out

def call_llama(question, system_prompt, max_new_tokens=300):
    cmd = [
        "python3", LLAMA_SCRIPT,
        "--model_dir", MODEL_DIR,
        "--mode", "int8",
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", "0.0",
        "--top_p", "1.0",
        "--system", system_prompt,
        "--question", question
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("LLAMA STDERR:", p.stderr, file=sys.stderr)
        raise RuntimeError("llama failed")
    return p.stdout.strip()

def main():
    os.makedirs(os.path.dirname(FINAL_OUT), exist_ok=True)
    raw = read_file(PROPOSER_OUT)
    draft_json = extract_json_like(raw)
    print("=== Draft JSON ===")
    print(draft_json[:400])
    func_code = read_file(SOURCE_SOL)
    verifier_prompt = render_verifier_prompt(func_code, draft_json)
    system = "너는 Solidity 보안 검증기다. 오직 최종 JSON 형식으로만 결과를 출력하라. 추가 설명은 금지. 증거가 부족하면 {\"label\":\"insufficient_evidence\"} 를 출력."
    final = call_llama(verifier_prompt, system_prompt=system, max_new_tokens=300)
    with open(FINAL_OUT, "w", encoding="utf-8") as f:
        f.write(final)
    print(">>> 최종 결과 저장 위치:", FINAL_OUT)
    print(final)

if __name__ == "__main__":
    main()
