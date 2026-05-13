#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-dl-exp3}"
CONDA_BASE="$(conda info --base)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi

echo "[1/4] create conda env: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python=3.10

PIP_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"
PY_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

echo "[2/4] install torch/torchvision (CUDA 12.8)"
"${PIP_BIN}" install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "[3/4] install notebook and experiment dependencies"
"${PIP_BIN}" install \
  matplotlib \
  scipy \
  jupyterlab \
  notebook \
  ipykernel \
  ipywidgets \
  pytorch-pretrained-biggan

echo "[4/4] register jupyter kernel"
"${PY_BIN}" -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo
echo "Done."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Kernel name: ${ENV_NAME}"
