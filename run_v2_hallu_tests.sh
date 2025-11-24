#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/home/mew/mev/llm/llama2-Chatv2.py"
MODEL_DIR="/home/mew/mev/llm/llama2"
PROMPT_FILE="/home/mew/mev/llm/prompts/prompts_cn.txt"
FACT_CTX="/home/mew/mev/llm/tests/input_fact.txt"

# 统一参数（v2：贪心解码最稳）
COMMON_ARGS=(--model_dir "$MODEL_DIR" --mode int8 --temperature 0 --top_p 1.0 --repetition_penalty 1.08 --max_new_tokens 300)

# 构造带“围栏”的 system（只能用 CONTEXT / 不足则 insufficient_evidence）
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
  printf "%s\n%s\n%s\n\n<CONTEXT>\n%s\n</CONTEXT>\n" "$base_sys" "$guard_cn" "$guard_kr" "$ctx"
}

# 提取文本中的数字（粗略）：
extract_numbers_py='
import re,sys
text=open(sys.argv[1],"r",encoding="utf-8",errors="ignore").read()
# 1) 去掉范围表达如 "3-5"（避免把 -5 当成单独的负数）
text=re.sub(r"\b\d+\s*-\s*\d+\b","",text)
# 2) 提取独立数字
nums=re.findall(r"(?<!\d)[+-]?\d+(?:\.\d+)?(?!\d)", text)
print("\n".join(sorted(set(nums))))
'

# 用例 1：有事实 ⇒ 不应该出现 insufficient_evidence；且输出中的数字必须都在 CONTEXT 里
echo "== Case 1: 有事实（应当成功总结且不编数字）"
SYS1="$(build_system_with_context "$FACT_CTX")"
OUT1="$(mktemp)"; SYSF1="$(mktemp)"
printf "%s" "$SYS1" > "$SYSF1"
python3 "$SCRIPT" "${COMMON_ARGS[@]}" \
  --system "$(cat "$SYSF1")" \
  --question "请严格只基于 <CONTEXT> 的材料，概括 3-5 条要点；若信息不足，输出 insufficient_evidence。不要编造来源或数字。" | tee "$OUT1"

# 断言 1a：不应出现 insufficient_evidence
if grep -qi "insufficient_evidence" "$OUT1"; then
  echo "[FAIL] 有事实却输出 insufficient_evidence"; exit 1
fi

# 断言 1b：输出数字必须出现在 CONTEXT 中
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
      echo "[FAIL] 检测到未被材料支撑的数字: $num"; exit 1
    fi
  done < "$OUTN"
fi
echo "[OK] Case 1 通过"

# 用例 2：无事实 ⇒ 必须输出 insufficient_evidence
echo "== Case 2: 无事实（必须拒答 insufficient_evidence）"
SYS2="$(build_system_with_context "/dev/null")"
OUT2="$(mktemp)"; SYSF2="$(mktemp)"
printf "%s" "$SYS2" > "$SYSF2"
python3 "$SCRIPT" "${COMMON_ARGS[@]}" \
  --system "$(cat "$SYSF2")" \
  --question "请严格只基于 <CONTEXT> 的材料总结要点；若信息不足，输出 insufficient_evidence。" | tee "$OUT2"

if grep -qi "insufficient_evidence" "$OUT2"; then
  echo "[OK] Case 2 通过"
else
  echo "[FAIL] 无事实但未输出 insufficient_evidence"; exit 1
fi

echo "== All tests passed =="
