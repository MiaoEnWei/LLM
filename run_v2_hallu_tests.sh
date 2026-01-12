#!/usr/bin/env bash
set -euo pipefail

# === Paths ===
SCRIPT="/home/mew/mev/llm/llama2-Chatv2.py"
MODEL_DIR="/home/mew/mev/llm/llama2"
PROMPT_FILE="/home/mew/mev/llm/prompts/prompts_cn.txt"
FACT_CTX="/home/mew/mev/llm/tests/input_fact.txt"

# Unified parameters (v2: Greedy decoding for maximum stability)
COMMON_ARGS=(--model_dir "$MODEL_DIR" --mode int8 --temperature 0 --top_p 1.0 --repetition_penalty 1.08 --max_new_tokens 300)

# Build system prompt with "Guardrails" (Restrict to CONTEXT / Output insufficient_evidence if lacking info)
build_system_with_context() {
  local ctx_file="$1"
  local base_sys
  base_sys="$(cat "$PROMPT_FILE" 2>/dev/null || true)"
  local guard_cn="你只能依据 <CONTEXT>...</CONTEXT> 的内容作答；不得引入上下文之外的数字或专有名词；若信息不足，必须输出 insufficient_evidence。"
  local guard_kr="컨텍스트(<CONTEXT>...</CONTEXT>) 안의 내용만 사용하여 답변하세요. 컨텍스트 밖의 숫자/고유명사는 금지. 정보가 부족하면 'insufficient_evidence'를 출력하세요."
  local ctx=""
  if [[ -f "$ctx_file" ]]; then
    ctx="$(cat "$ctx_file")"
  fi
  # Merging base system prompt, language guardrails, and context
  printf "%s\n%s\n%s\n\n<CONTEXT>\n%s\n</CONTEXT>\n" "$base_sys" "$guard_cn" "$guard_kr" "$ctx"
}

# Extract numbers from text (Rough estimation via Python):
extract_numbers_py='
import re,sys
text=open(sys.argv[1],"r",encoding="utf-8",errors="ignore").read()
# 1) Remove range expressions like "3-5" (avoids treating -5 as a separate negative number)
text=re.sub(r"\b\d+\s*-\s*\d+\b","",text)
# 2) Extract independent numbers
nums=re.findall(r"(?<!\d)[+-]?\d+(?:\.\d+)?(?!\d)", text)
print("\n".join(sorted(set(nums))))
'

# Case 1: Facts present => Should NOT output insufficient_evidence; output numbers must exist in CONTEXT
echo "== Case 1: Fact available (Should summarize successfully without hallucinating numbers)"
SYS1="$(build_system_with_context "$FACT_CTX")"
OUT1="$(mktemp)"; SYSF1="$(mktemp)"
printf "%s" "$SYS1" > "$SYSF1"
python3 "$SCRIPT" "${COMMON_ARGS[@]}" \
  --system "$(cat "$SYSF1")" \
  --question "Summarize 3-5 key points based strictly on the <CONTEXT> material. If information is insufficient, output insufficient_evidence. Do not fabricate sources or numbers." | tee "$OUT1"

# Assertion 1a: Should not see insufficient_evidence
if grep -qi "insufficient_evidence" "$OUT1"; then
  echo "[FAIL] Facts were provided, but 'insufficient_evidence' was output"; exit 1
fi

# Assertion 1b: All numbers in output must be present in the CONTEXT
OUTN="$(mktemp)"; CTXN="$(mktemp)"
python3 - <<PY "$OUT1" > "$OUTN"
$extract_numbers_py
PY
python3 - <<PY "$FACT_CTX" > "$CTXN"
$extract_numbers_py
PY
if [[ -s "$OUTN" ]]; then
  while read -r num; do
    if ! grep -q -F "$num" "$CTXN"; then
      echo "[FAIL] Detected number not supported by material: $num"; exit 1
    fi
  done < "$OUTN"
fi
echo "[OK] Case 1 Passed"

# Case 2: No facts => Must output insufficient_evidence
echo "== Case 2: No facts (Must refuse with 'insufficient_evidence')"
SYS2="$(build_system_with_context "/dev/null")"
OUT2="$(mktemp)"; SYSF2="$(mktemp)"
printf "%s" "$SYS2" > "$SYSF2"
python3 "$SCRIPT" "${COMMON_ARGS[@]}" \
  --system "$(cat "$SYSF2")" \
  --question "Summarize key points based strictly on the <CONTEXT> material. If info is insufficient, output insufficient_evidence." | tee "$OUT2"

if grep -qi "insufficient_evidence" "$OUT2"; then
  echo "[OK] Case 2 Passed"
else
  echo "[FAIL] No facts provided, but 'insufficient_evidence' was not output"; exit 1
fi

echo "== All tests passed =="