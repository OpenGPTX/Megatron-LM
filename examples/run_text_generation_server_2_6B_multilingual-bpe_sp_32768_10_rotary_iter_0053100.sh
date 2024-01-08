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
       --load /p/scratch/opengptx-elm/ali5/opengpt/megatron-lm/2023-07-27_17-52-53/output_dir/2_6B_multilingual-bpe_sp_32768_10_rotary.sh/checkpoints \
       --tokenizer-model /p/scratch/opengptx-elm/data/datasources_opgptx/data_quality_experiments_datasets/ablations_studies/multilingual/70B_EQW_10/tokenizer_training/bpe/sp/32768_10/bpe_tokenizer.model \
       --tokenizer-type OpenGPTX-SPTokenizer \
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

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000"


bash -c "$LAUNCHER $CMD" 
