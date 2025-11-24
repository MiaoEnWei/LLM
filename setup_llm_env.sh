#!/usr/bin/env bash
set -euo pipefail

# === 根目录 ===
ROOT="/home/mew/mev/llm"
ENV_DIR="$ROOT/.venv"

echo ">>> [1/7] 准备系统组件"
sudo apt-get update -y
sudo apt-get install -y python3-venv git build-essential

echo ">>> [2/7] 创建并激活全新 venv: $ENV_DIR"
mkdir -p "$ROOT"
rm -rf "$ENV_DIR"
python3 -m venv "$ENV_DIR"
# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

echo ">>> [3/7] 升级 pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo ">>> [4/7] 安装 PyTorch (CUDA 12.1 版，适配 3080 Ti)"
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121

echo ">>> [5/7] 安装通用依赖（固定版本）"
pip install \
  "transformers==4.44.2" "tokenizers==0.19.1" accelerate safetensors sentencepiece \
  "bitsandbytes==0.43.3" \
  "scikit-learn==1.4.2" \
  "rank_bm25==0.2.2" \
  psutil numpy

# （可选）NLTK：用于句子切分回退
pip install nltk
python - <<'PY'
import nltk
for p in ("punkt","punkt_tab"):
    try: nltk.download(p, quiet=True)
    except Exception: pass
print("NLTK data ready.")
PY

echo ">>> [6/7] 版本与 GPU 自检"
python - <<'PY'
import torch, transformers, sys
print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__, "CUDA:", torch.cuda.is_available(), "Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
print("Transformers:", transformers.__version__)
PY
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi 不可用（无独显或驱动未装）"

echo ">>> [7/7] 便捷别名（追加到 ~/.bashrc）"
ALIAS_LINE='alias llm-venv="source /home/mew/mev/llm/.venv/bin/activate"'
grep -qxF "$ALIAS_LINE" "$HOME/.bashrc" || echo "$ALIAS_LINE" >> "$HOME/.bashrc"

echo "完成。使用前执行：source \"/home/mew/mev/llm/.venv/bin/activate\"  或  llm-venv"
