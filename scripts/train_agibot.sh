#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -x "${PROJECT_ROOT}/env/bin/python" ]; then
  PYTHON="${PROJECT_ROOT}/env/bin/python"
else
  PYTHON=python
fi

export TOKENIZERS_PARALLELISM=false

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
else
  NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Using NPROC_PER_NODE=$NPROC_PER_NODE GPUs"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-0.0.0.0}
MASTER_PORT=${MASTER_PORT:-62500}

CONFIG=${1:-${PROJECT_ROOT}/configs/vla/agibot_pick_block.yaml}
shift 2>/dev/null

$PYTHON -m torch.distributed.run \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  ${PROJECT_ROOT}/tasks/vla/train_lingbotvla.py "$CONFIG" "$@" 2>&1 | tee ${PROJECT_ROOT}/log.txt
