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
 --load /p/scratch/opengptx-elm/ali5/opengpt/megatron-lm/2023-07-27_17-52-53/output_dir/2_6B_monolingual_eng-bpe_sp_32768_10_rotary.sh/checkpoints \
 --out-seq-length 2048  \
 --temperature 0.8  \
 --top_p 0.5  \
 --tensor-model-parallel-size 2 \
 --pipeline-model-parallel-size 1 \
 --max-tokens-to-oom=300000 \
 --num-layers 32 \
 --hidden-size 2560 \
 --num-attention-heads 32 \
 --seq-length 2048 \
 --max-position-embeddings 2048 \
 --micro-batch-size 5 \
 --global-batch-size 480 \
 --train-samples 25_488_281 \
 --tokenizer-type OpenGPTX-SPTokenizer \
 --tokenizer-model /p/scratch/opengptx-elm/data/datasources_opgptx/data_quality_experiments_datasets/ablations_studies/monolingual_en/70B_10/tokenizer_training/bpe/sp/32768_10/bpe_tokenizer.model \
 --init-method-std 0.02 \
 --bf16 \
 --seed 42 \
 --reset-attention-mask \
 --reset-position-ids \
 --use-flash-attn \
 --no-position-embedding \
 --position-embedding-type rotary \
 --optimizer adam \
 --adam-beta1 0.9 \
 --adam-beta2 0.95 \
 --adam-eps 1e-8 \
 --lr 1.6e-4 \
 --min-lr 1.6e-5 \
 --lr-decay-style cosine \
 --lr-decay-samples 22_089_843 \
 --lr-warmup-samples 31_860 \
 --clip-grad 1.0 \
 --weight-decay 1e-1 \
 --use-distributed-optimizer \
 --log-interval 100 \
 --log-memory-to-tensorboard \
 --log-world-size-to-tensorboard \
 --save-interval 3000 \
 --eval-interval 3000 \
 --eval-iters 1 \
 --tensorboard-dir /p/scratch/opengptx-elm/ali5/opengpt/megatron-lm/2023-07-27_17-52-53/output_dir/2_6B_monolingual_eng-bpe_sp_32768_10_rotary.sh/tensorboard \
 --tensorboard-queue-size 5 \
 --log-timers-to-tensorboard \
 --log-batch-size-to-tensorboard \
 --log-validation-ppl-to-tensorboard \
 --num-workers 11 \
 --data-impl mmap \
 --distributed-backend nccl \
 --load /p/scratch/opengptx-elm/ali5/opengpt/megatron-lm/2023-07-27_17-52-53/output_dir/2_6B_monolingual_eng-bpe_sp_32768_10_rotary.sh/checkpoints \
"

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000"


bash -c "$LAUNCHER $CMD" 
