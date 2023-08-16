#!/bin/bash

set -x -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json
export NCCL_ASYNC_ERROR_HANDLING=1
export UCX_RC_TIMEOUT=4s
export NCCL_DEBUG=INFO
export MAX_JOBS=$SLURM_JOB_CPUS_PER_NODE

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-${(%):-%x}}" )" &> /dev/null && pwd )


export CMD=" \
       $SCRIPT_DIR/../tools/run_text_generation_server.py \
       --load /p/project/opengptx-elm/thellmann1/opengpt_2023/gpt345M \
       --vocab-file /p/project/opengptx-elm/thellmann1/opengpt_2023/gpt345M/gpt2-vocab.json \
       --merge-file /p/project/opengptx-elm/thellmann1/opengpt_2023/gpt345M/gpt2-merges.txt \
       --tokenizer-type GPT2BPETokenizer \
       --tensor-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --max-position-embeddings 1024 \
       --fp16 \
       --micro-batch-size 8 \
       --seq-length 1024 \
       --max-tokens-to-oom 300000 \
       --out-seq-length 1024 \
       --temperature 0.8 \
       --top_p 0.5
       "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000"


bash -c "$LAUNCHER $CMD" 
