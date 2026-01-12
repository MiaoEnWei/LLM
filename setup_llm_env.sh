#!/usr/bin/env bash
set -euo pipefail

# === Root Directory ===
ROOT="/home/mew/mev/llm"
ENV_DIR="$ROOT/.venv"

echo ">>> [1/7] Preparing system components"
sudo apt-get update -y
sudo apt-get install -y python3-venv git build-essential

echo ">>> [2/7] Creating and activating fresh venv: $ENV_DIR"
mkdir -p "$ROOT"
rm -rf "$ENV_DIR"
python3 -m venv "$ENV_DIR"
# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

echo ">>> [3/7] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo ">>> [4/7] Installing PyTorch (CUDA 12.1 version, compatible with 3080 Ti)"
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121

echo ">>> [5/7] Installing general dependencies (pinned versions)"
pip install \
  "transformers==4.44.2" "tokenizers==0.19.1" accelerate safetensors sentencepiece \
  "bitsandbytes==0.43.3" \
  "scikit-learn==1.4.2" \
  "rank_bm25==0.2.2" \
  psutil numpy

# (Optional) NLTK: used for sentence splitting fallback
pip install nltk
python - <<'PY'
import nltk
for p in ("punkt","punkt_tab"):
    try: nltk.download(p, quiet=True)
    except Exception: pass
print("NLTK data ready.")
PY

echo ">>> [6/7] Version and GPU self-check"
python - <<'PY'
import torch, transformers, sys
print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__, "CUDA:", torch.cuda.is_available(), "Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
print("Transformers:", transformers.__version__)
PY
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi unavailable (no GPU or driver not installed)"

echo ">>> [7/7] Convenience alias (append to ~/.bashrc)"
ALIAS_LINE='alias llm-venv="source /home/mew/mev/llm/.venv/bin/activate"'
grep -qxF "$ALIAS_LINE" "$HOME/.bashrc" || echo "$ALIAS_LINE" >> "$HOME/.bashrc"

echo "Finished. Before use, run: source \"/home/mew/mev/llm/.venv/bin/activate\" or llm-venv"