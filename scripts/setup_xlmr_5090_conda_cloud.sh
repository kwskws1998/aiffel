#!/usr/bin/env bash
# Conda bootstrap for four-GPU XLM-R large training on RTX 5090 cloud nodes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-va_gaze}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-4}"
PRELOAD_MODELS="${PRELOAD_MODELS:-1}"
DATA_DIR="${DATA_DIR:-./data}"
FILTERED_DATA_DIR="${FILTERED_DATA_DIR:-./data_no_iemocap}"
ET2_CHECKPOINT="${ET2_CHECKPOINT:-./checkpoints/et_predictor2_seed123}"
DATA_ZIP_FILE_ID="${DATA_ZIP_FILE_ID:-1xXM32nva_4I3EAVAOrQ84L16f-LjsJbj}"
DATA_ZIP_SHA256="${DATA_ZIP_SHA256:-5db750ededfd9717dcca465b34fd7e6c348e50e563ad2c0814c458b04441e81d}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  printf 'Deactivate the active virtualenv before running this script: %s\n' "$VIRTUAL_ENV" >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  printf 'Conda is not available in PATH.\n' >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is unavailable. Start from an NVIDIA RTX 5090 cloud image.\n' >&2
  exit 2
fi

nvidia-smi

if ! conda run -n "$CONDA_ENV" python -V >/dev/null 2>&1; then
  conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION" pip
fi

PYTHON_BIN="$(conda run -n "$CONDA_ENV" python -c 'import sys; print(sys.executable)' | tail -n 1 | tr -d '\r')"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install --upgrade "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"

TEMP_REQUIREMENTS="$(mktemp /tmp/va_gaze_5090_requirements_no_torch.XXXXXX)"
trap 'rm -f "$TEMP_REQUIREMENTS"' EXIT
sed '/^torch==/d' requirements.txt > "$TEMP_REQUIREMENTS"
"$PYTHON_BIN" -m pip install -r "$TEMP_REQUIREMENTS"

export PYTHON_BIN
export SKIP_DEPS=1
export WITH_ET1=0
export DATA_DIR
export FILTERED_DATA_DIR
export ET2_CHECKPOINT
export DATA_ZIP_FILE_ID
export DATA_ZIP_SHA256

bash install.sh

"$PYTHON_BIN" filter_datasets.py \
  --input-dir "$DATA_DIR" \
  --output-dir "$FILTERED_DATA_DIR" \
  --exclude IEMOCAP

if [[ "$PRELOAD_MODELS" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer

AutoTokenizer.from_pretrained("xlm-roberta-large")
AutoModel.from_pretrained("xlm-roberta-large")
RobertaTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
RobertaModel.from_pretrained("roberta-base")
PY
fi

export REQUIRED_GPU_COUNT
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import pandas as pd
import torch

required_gpu_count = int(os.environ["REQUIRED_GPU_COUNT"])
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() >= required_gpu_count, (
    f"Expected at least {required_gpu_count} GPUs, found {torch.cuda.device_count()}"
)
assert str(torch.version.cuda).startswith("12.8"), torch.version.cuda
for device_index in range(required_gpu_count):
    capability = torch.cuda.get_device_capability(device_index)
    assert capability >= (12, 0), (device_index, capability)
    print(device_index, torch.cuda.get_device_name(device_index), capability)

checkpoint_base = Path(os.environ["ET2_CHECKPOINT"])
checkpoint_candidates = [
    checkpoint_base,
    Path(str(checkpoint_base) + ".safetensors"),
    Path(str(checkpoint_base) + ".pt"),
    Path(str(checkpoint_base) + ".bin"),
]
assert any(path.is_file() for path in checkpoint_candidates), checkpoint_candidates
for filename in ("full_dataset_fold1.csv", "full_dataset_fold2.csv"):
    path = Path(os.environ["FILTERED_DATA_DIR"]) / filename
    assert path.is_file(), path
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    assert not frame["dataset_of_origin"].str.contains("IEMOCAP", case=False).any()

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("four_gpu_5090_environment=ok")
PY

printf 'Environment ready. Activate with: conda activate %s\n' "$CONDA_ENV"
