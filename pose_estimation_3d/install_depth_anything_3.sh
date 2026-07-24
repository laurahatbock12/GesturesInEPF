#!/usr/bin/env bash
# Install Depth Anything 3 (https://github.com/ByteDance-Seed/Depth-Anything-3) into its own
# conda environment, isolated from the other pose-estimation environments in this project.
#
# Usage:
#   bash pose_estimation_3d/install_depth_anything_3.sh [repo_dir]
#
# repo_dir defaults to ~/Documents/Depth-Anything-3
set -euo pipefail

ENV_NAME="da3"
PYTHON_VERSION="3.10"
REPO_DIR="${1:-$HOME/Documents/Depth-Anything-3}"
REPO_URL="https://github.com/ByteDance-Seed/Depth-Anything-3.git"

# Pinned versions: this machine's driver reports CUDA 12.4 (see `nvidia-smi`), so we pin the
# plain-PyPI torch/torchvision/xformers combo whose CUDA runtime wheels line up with that
# (torch 2.5.1 bundles CUDA-12.4 nvidia-* runtime packages, and xformers 0.0.29.post1 is the
# last xformers release pinned to torch==2.5.1). Newer torch/xformers releases only ship
# CUDA-12.6+ wheels, which this driver cannot run.
TORCH_VERSION="2.5.1"
TORCHVISION_VERSION="0.20.1"
XFORMERS_VERSION="0.0.29.post1"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists, reusing it."
else
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

# Deliberately avoid `conda activate` / relying on PATH: some interactive shells on this
# machine put another env's bin/ ahead of the activated one, which silently sends every
# subsequent `pip`/`python` call to the wrong environment. Absolute paths sidestep that.
ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"
PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

"$PIP" install --upgrade pip
"$PIP" install "numpy<2"
"$PIP" install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "xformers==$XFORMERS_VERSION"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already present at $REPO_DIR, leaving it as-is (git pull manually to update)."
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

"$PIP" install -e "$REPO_DIR"

echo
echo "Sanity check:"
"$PY" -c "
import torch, xformers
from depth_anything_3.api import DepthAnything3
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('xformers', xformers.__version__)
print('depth_anything_3 import OK')
"

echo
echo "Pre-downloading the default checkpoint (depth-anything/DA3METRIC-LARGE) from Hugging Face..."
"$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('depth-anything/DA3METRIC-LARGE')
print('Checkpoint cached.')
"

echo
echo "Done. Activate with: conda activate $ENV_NAME"
echo "(If 'conda activate $ENV_NAME' doesn't put the right python first on PATH in your shell,"
echo " run 'which python' to check, or just call $PY directly.)"
echo "Then run, e.g.:"
echo "  python pose_estimation_3d/estimate_monocular_depth.py --process_reference_videos --project_folder /media1/data/andy/laura_disk/EPF_PL_DA"
