#!/bin/bash
# Environment setup for OSU HPC
# Before running, load modules and activate venv:
#   module load python/3.10 cuda/12.1
#   python -m venv .venv && source .venv/bin/activate
#   bash env_setup.sh

set -e

pip install --upgrade pip

# PyTorch pinned to match CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# General deps (latest ok)
pip install scikit-learn pyyaml tqdm tensorboard networkx numpy

# PyG
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
	    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo "Done. Verify: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
