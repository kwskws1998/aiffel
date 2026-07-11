#!/usr/bin/env bash
# Conda bootstrap for DistilBERT experiments on a rented Linux NVIDIA GPU.
# Example: DATA_ZIP_FILE_ID=... bash scripts/setup_distilbert_conda_cloud.sh

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: bash scripts/setup_distilbert_conda_cloud.sh" \
    "" \
    "Environment variables:" \
    "  CONDA_ENV=va_gaze       Conda environment name" \
    "  PYTHON_VERSION=3.10     Python version for the environment" \
    "  TORCH_VERSION=2.2.2     PyTorch version" \
    "  CUDA_RUNTIME=12.1       Conda pytorch-cuda runtime; use 11.8 if needed" \
    "  DATA_ZIP_FILE_ID=...    Authorized Google Drive zip id" \
    "  DATA_ZIP_URL=...        Authorized Google Drive share URL" \
    "  DATA_ZIP_SHA256=...     Optional expected checksum" \
    "  DATA_DIR=./data         Prepared fold directory" \
    "  ET2_CHECKPOINT=...      ET2 checkpoint base path" \
    "  PRELOAD_MODELS=0        Skip DistilBERT/RoBERTa cache preloading"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-va_gaze}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.2.2}"
CUDA_RUNTIME="${CUDA_RUNTIME:-12.1}"
PRELOAD_MODELS="${PRELOAD_MODELS:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not available in PATH." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; start from an NVIDIA GPU cloud image." >&2
  exit 2
fi

nvidia-smi

if ! conda run -n "$CONDA_ENV" python -V >/dev/null 2>&1; then
  conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION" pip
fi

if ! conda run -n "$CONDA_ENV" python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  conda install -y -n "$CONDA_ENV" \
    "pytorch=$TORCH_VERSION" \
    "pytorch-cuda=$CUDA_RUNTIME" \
    -c pytorch -c nvidia
fi

PYTHON_BIN="$(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.executable)' | tail -n 1 | tr -d '\r')"
"$PYTHON_BIN" -m pip install -U pip setuptools wheel

TEMP_REQUIREMENTS="$(mktemp /tmp/va_gaze_requirements_no_torch.XXXXXX)"
trap 'rm -f "$TEMP_REQUIREMENTS"' EXIT
sed '/^torch==/d' requirements.txt > "$TEMP_REQUIREMENTS"
"$PYTHON_BIN" -m pip install -r "$TEMP_REQUIREMENTS"

export PYTHON_BIN
export SKIP_DEPS=1
export WITH_ET1="${WITH_ET1:-0}"
export ET2_CHECKPOINT="${ET2_CHECKPOINT:-./checkpoints/et_predictor2_seed123}"
export DATA_DIR="${DATA_DIR:-./data}"

bash install.sh

if [[ "$PRELOAD_MODELS" == "1" ]]; then
  "$PYTHON_BIN" -c 'from transformers import AutoModel,AutoTokenizer,RobertaModel,RobertaTokenizer; AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased"); AutoModel.from_pretrained("distilbert-base-multilingual-cased"); RobertaTokenizer.from_pretrained("roberta-base", add_prefix_space=True); RobertaModel.from_pretrained("roberta-base")'
fi

"$PYTHON_BIN" -c 'import torch,transformers,gdown; assert torch.cuda.is_available(); print("torch", torch.__version__); print("torch_cuda", torch.version.cuda); print("transformers", transformers.__version__); print("gpu", torch.cuda.get_device_name(0))'

CONDA_BASE="$(conda info --base)"
echo "Conda cloud environment ready."
echo "Activate with: source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $CONDA_ENV"
echo "Train with: REQUIRE_CUDA=1 bash scripts/run_distilbert_experiments.sh"
