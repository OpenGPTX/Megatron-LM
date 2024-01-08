#!/bin/bash

set -x -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=60234
export MAX_JOBS=$SLURM_JOB_CPUS_PER_NODE

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-${(%):-%x}}" )" &> /dev/null && pwd )


export CMD=" \
       $SCRIPT_DIR/../tools/run_text_generation_server.py \
       --load /beegfs/p_gptx/tokenizer_study/cp_2_6B_iter_0053100/checkpoints/2_6B_monolingual_eng-bpe_hf_32768_10_rotary.sh/checkpoints \
       --tokenizer-model /beegfs/p_gptx/tokenizer_study/2_6B_tokenizer_models/2_6B_monolingual_eng-bpe_hf_32768_10_rotary.sh/tokenizer/iter_0053100/tokenizer.json \
       --tokenizer-type OpenGPTX-HFTokenizer \
       --pipeline-model-parallel-size 1 \
       --tensor-model-parallel-size 2 \
       --num-layers 32  \
       --hidden-size 2560  \
       --num-attention-heads 32  \
       --max-position-embeddings 2048  \
       --bf16  \
       --micro-batch-size 5  \
       --seq-length 2048  \
       --out-seq-length 2048  \
       --temperature 0.8  \
       --top_p 0.5  \
       --seed 42 \
       --position-embedding-type rotary \
       --no-position-embedding \
       "

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000"


bash -c "$LAUNCHER $CMD"

