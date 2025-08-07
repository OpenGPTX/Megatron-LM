#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=7B_EU24
#SBATCH --account=opengptx-elm
#SBATCH --partition=booster
#SBATCH --nodes=128
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=23:59:50
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.fromm@iais.fraunhofer.de
#SBATCH --output=/p/project/opengptx-elm/fromm1/OpenGPTX-Megatron-LM-Setup/sample_scripts/juwels/%x-%j.out           # output file name
#SBATCH --error=/p/project/opengptx-elm/fromm1/OpenGPTX-Megatron-LM-Setup/sample_scripts/juwels/%x-%j.err            # error file name
# explicitly setting srun environment variable to inherit from SBATCH
#SBATCH --array=0-100%1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}


# Enable logging
set -x -e
echo "START TIME: $(date)"

# Get location of setup scripts
if [ -z "${SETUP_SCRIPTS_DIR:-}" ]; then
    echo "SETUP_SCRIPTS_DIR is not set. Trying to find it as a parent directory of this script."
    SCRIPT_PATH=$(scontrol show job ${SLURM_JOB_ID-} | awk -F= '/Command=/{print $2}')
    if [[ "$SCRIPT_PATH" == *"OpenGPTX-Megatron-LM-Setup"* ]]; then
        export SETUP_SCRIPTS_DIR="${SCRIPT_PATH%OpenGPTX-Megatron-LM-Setup/*}OpenGPTX-Megatron-LM-Setup/"
        echo "SETUP_SCRIPTS_DIR set to ${SETUP_SCRIPTS_DIR}"
    else
        echo "Could not find directory OpenGPTX-Megatron-LM-Setup in path, please set SETUP_SCRIPTS_DIR manually and try again." 1>&2 
        exit 1
    fi
fi
source "$SETUP_SCRIPTS_DIR"/setupworkenv.sh|| exit 1 

mkdir -p "${OUTPUT_DIR:-.}"

# Code Base path
MEGATRON_LM_REPO="$INSTALLATION_DIR"/Megatron-LM

#### Input  data ####
# TOKENIZER_MODEL_FILE=/beegfs/p_gptx/datasources_opgptx/data_quality_experiments_datasets/ablations_studies/language_share/tokenizer/unigram_EQW/unigram_tokenizer.model
TOKENIZER_MODEL_FILE=/p/scratch/opengptx-elm/data/trainings/7B_EU24/tokenizer/unigram_tokenizer.model
# DATA_PATH=/beegfs/p_gptx/test_setup_data/oscar_text_document
DATA_PATH=/p/scratch/opengptx-elm/data/trainings/7B_EU24/phase_4/train

#### Output paths ####
[[ -z "${DATA_OUTPUT_PATH:-}" ]] && DATA_OUTPUT_PATH="/p/scratch/opengptx-elm/fromm1/output/7B_fw_decay"
CHECKPOINT_PATH="$DATA_OUTPUT_PATH"/checkpoints
TENSORBOARD_PATH="$DATA_OUTPUT_PATH"/tensorboard
#CODECARBON_PATH="$DATA_OUTPUT_PATH"/codecarbon
#CACHE_DIR="$DATA_OUTPUT_PATH/.cache"
LOGS_PATH="$DATA_OUTPUT_PATH"/logs
TORCHELASTIC_ERROR_FILE=$LOGS_PATH/torch_distribute_error.txt

mkdir -p $LOGS_PATH

# copy this batch script into log directory for reproducibility
if [ -e "$0" ]; then
    cp -p "$0" "$LOGS_PATH/batch-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.sh"
fi


#### Environment variables ####
[[ -z "${LOAD_CHECKPOINTS:-}" ]] && LOAD_CHECKPOINTS=true   # LOAD_CHECKPOINTS=true may be set outside
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CXX=g++
export CC=gcc
# Necessary for some Megatron-LM settings. We set it all the time just
# to be safe.
export CUDA_DEVICE_MAX_CONNECTIONS=1
# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=20
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO

# Get IP for hostname.
MASTER_ADDR="$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)"i
MASTER_PORT=6000

cd "$MEGATRON_LM_REPO"
CLEAN_PREV_JIT_BUILD=0

rm -f megatron/fused_kernels/build/lock
((CLEAN_PREV_JIT_BUILD)) && rm -rf megatron/fused_kernels/{build,__pycache__}



MODEL_SIZE=7
##### Parallel model layouting #####
# This is an example configuration, not an optimized one

GPUS_PER_NODE=4 # Minimum required number of GPUs: PP_SIZE * TP_SIZE 
NNODES="$SLURM_JOB_NUM_NODES"
PP_SIZE=2
TP_SIZE=2 # NLAYERS must be a multiple of PP_SIZE here

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024

NLAYERS=32
NHIDDEN=4096
FFNHIDDEN=13440 #corresponds to number of groups of 2
NHEADS=32
SEQ_LEN=4096
VOCAB_SIZE=250680

SAVE_INTERVAL=225
LOG_INTERVAL=10
EVAL_INTERVAL=450

TRAIN_SAMPLES=732_421_875
# TRAIN_TOKENS=3_000_000_000_000

LR_DECAY_SAMPLES=732_421_875  # Corresponds to 3T tokens

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 3.0e-5 \
    --min-lr 1.5e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --use-distributed-optimizer \
    "
		
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --t5-swiglu \
    --hidden-size $NHIDDEN \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --ffn-hidden-size $FFNHIDDEN \
    --group-query-attention \
    --num-query-groups 2 \
    --num-attention-heads $NHEADS \
    --normalization RMSNorm \
    --disable-bias-linear \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type OpenGPTX-SPTokenizer \
    --position-embedding-type rope \
    --tokenizer-model $TOKENIZER_MODEL_FILE \
    --bf16 \
    --seed 42 \
    --recompute-activations \
    --use-flash-attn \
    --sequence-parallel \
    --init-method-std 0.0158 \
    --data-cache-path /p/scratch/opengptx-elm/data/trainings/7B_EU24/part_3_fw/index-cache
    $OPTIMIZER_ARGS 
    "
	
OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters 5 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-world-size-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-num-zeros-in-grad \
    --exit-signal-handler \
"

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    "

export CMD=" \
    ./pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 990,5,5 \
    --distributed-backend nccl \
    --num-workers 11 \
    "
	
if [ "$LOAD_CHECKPOINTS" = true ] ; then
    export CMD="$CMD\
        --load $CHECKPOINT_PATH \
        "
fi

echo $CMD

srun --jobid $SLURM_JOB_ID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'  2>&1 \
    | tee -a "$LOGS_PATH"/main_log.txt

echo "END TIME: $(date)"