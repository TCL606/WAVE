#!/bin/bash
cd "$(cd $(dirname $0); pwd)/.."
echo "All parameters: $@"

export HF_HOME="/mnt/sh/mmvision/home/changlitang/huggingface"

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_IB_TIMEOUT=24
export NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME=bond1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

export TORCH_DISTRIBUTED_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=86400 # 4H waitting
export TORCH_NCCL_ENABLE_MONITORING=0 # waitting for Infinity

export ACCELERATE_BACKEND=nccl
export NCCL_TIMEOUT_MINS=30

DATASET=1
MODEL=1
MODEL_BASE=1
LR=2e-5
BS=1
ACCUM_STEPS=1
RUN_NAME=debug
DEEPSPEED=0
TRAIN_LLM=False
TRAIN_PROJ=False
TRAIN_ENC=False
TRAIN_AUDIO=False
TRAIN_QFORMER=False
EPOCH=1

MAX_PIXELS=176400
MIN_PIXELS=784

SAVE_STEPS=1000

MIN_FRAMES=1
MAX_FRAMES=128
INTERVAL=0.2

USE_LORA=False
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05
LORA_CKPT=No

NUM_WORKER=8
TRAIN_CLASSIFY=False
CLASSIFY_DIM=24

CLASSIFY_TYPE=last_layer
USE_TYPE_SAMPLER=False

USE_BEATS=False
BEATS_ONLY=False
TUNE_BEATS_PROJ=False
GPU_NUM=8

mkdir -p output
mkdir -p dataset

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --model_base) MODEL_BASE="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --run_name) RUN_NAME="$2"; shift ;;
        --bs) BS="$2"; shift ;;
        --accum_steps) ACCUM_STEPS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --deepspeed) DEEPSPEED="$2"; shift ;;
        --train_llm) TRAIN_LLM=True ;;
        --train_proj) TRAIN_PROJ=True ;;
        --train_enc) TRAIN_ENC=True ;;
        --train_audio) TRAIN_AUDIO=True ;;
        --train_qformer) TRAIN_QFORMER=True ;;
        --max_pixels) MAX_PIXELS="$2"; shift ;;
        --min_pixels) MIN_PIXELS="$2"; shift ;;
        --epoch) EPOCH="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --min_frames) MIN_FRAMES="$2"; shift ;;
        --max_frames) MAX_FRAMES="$2"; shift ;;
        --interval) INTERVAL="$2"; shift ;;
        --use_lora) USE_LORA=True ;;
        --lora_r) LORA_R="$2"; shift ;;
        --lora_alpha) LORA_ALPHA="$2"; shift ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift ;;
        --lora_ckpt) LORA_CKPT="$2"; shift ;;
        --num_worker) NUM_WORKER="$2"; shift ;;
        --train_classify) TRAIN_CLASSIFY=True ;;
        --classify_dim) CLASSIFY_DIM="$2"; shift ;;
        --classify_type) CLASSIFY_TYPE="$2"; shift ;;
        --use_type_sampler) USE_TYPE_SAMPLER=True ;;
        --use_beats) USE_BEATS=True ;;
        --beats_only) BEATS_ONLY=True ;;
        --tune_beats_proj) TUNE_BEATS_PROJ=True ;;
        --gpu_num) GPU_NUM="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

torchrun --nproc_per_node=$GPU_NUM --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=46001 \
    qwenvl/train/train_qwen.py \
        --deepspeed scripts/zero${DEEPSPEED}.json \
        --model_name_or_path "$MODEL" \
        --dataset_use $DATASET \
        --tune_mm_vision $TRAIN_ENC \
        --tune_mm_mlp $TRAIN_PROJ \
        --tune_mm_llm $TRAIN_LLM \
        --bf16 \
        --output_dir output/$RUN_NAME \
        --num_train_epochs $EPOCH \
        --per_device_train_batch_size $BS \
        --gradient_accumulation_steps $ACCUM_STEPS \
        --max_pixels $MAX_PIXELS \
        --min_pixels $MIN_PIXELS \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps $SAVE_STEPS \
        --save_total_limit 100 \
        --learning_rate $LR \
        --weight_decay 0 \
        --warmup_ratio 0.03 \
        --max_grad_norm 1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 131072 \
        --gradient_checkpointing True \
        --dataloader_num_workers $NUM_WORKER \
        --run_name $RUN_NAME \
        --video_min_frames $MIN_FRAMES \
        --video_max_frames $MAX_FRAMES \
        --base_interval $INTERVAL \
        --model_base $MODEL_BASE \
        --use_lora $USE_LORA \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_ckpt $LORA_CKPT \
        --tune_mm_audio $TRAIN_AUDIO \
        --tune_mm_qformer $TRAIN_QFORMER \
        --report_to tensorboard \
        --train_classify $TRAIN_CLASSIFY \
        --classify_dim $CLASSIFY_DIM \
        --classify_type $CLASSIFY_TYPE \
        --use_type_sampler $USE_TYPE_SAMPLER \
        --use_beats $USE_BEATS \
        --beats_only $BEATS_ONLY \
        --tune_beats_proj $TUNE_BEATS_PROJ \
