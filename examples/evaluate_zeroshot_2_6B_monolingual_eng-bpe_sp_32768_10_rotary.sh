#!/bin/bash


export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="${MASTER_ADDR}i"
MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
MASTER_PORT=6000


DISTRIBUTED_ARGS="--nproc_per_node 2 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
                  --rdzv_backend c10d"


python -u -m torch.distributed.run $DISTRIBUTED_ARGS ./tasks/main.py \
               --task "LAMBADA" \
               --valid-data /p/project/opengptx-elm/thellmann1/opengpt_2023/data/bflm/lambada_test.jsonl \
               --strict-lambada \
               --tokenizer-type OpenGPTX-SPTokenizer \
               --tokenizer-model /p/scratch/opengptx-elm/data/datasources_opgptx/data_quality_experiments_datasets/ablations_studies/monolingual_en/70B_10/tokenizer_training/bpe/sp/32768_10/bpe_tokenizer.model \
               --load /p/scratch/opengptx-elm/ali5/opengpt/megatron-lm/2023-07-27_17-52-53/output_dir/2_6B_monolingual_eng-bpe_sp_32768_10_rotary.sh/checkpoints \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 1 \
               --no-position-embedding \
               --position-embedding-type rotary \
               --num-layers 32 \
               --hidden-size 2560 \
               --num-attention-heads 32 \
               --micro-batch-size 5 \
               --global-batch-size 480 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --log-interval 10 \
               --bf16
