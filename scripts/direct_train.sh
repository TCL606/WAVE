#!/bin/bash

cd "$(cd $(dirname $0); pwd)"

data=/mnt/sh/mmvision/home/changlitang/preprocess_dataset/msrvtt_train_av.json

bash train.sh \
    --interval 0.5 \
    --run_name debug \
    --dataset $data \
    --lr 2e-5 --bs 1 --epoch 1 \
    --max_frames 128 --save_steps 50000 \
    --model /mnt/sh/mmvision/home/changlitang/avllm_embed/output/sft_beatsOnly_ac+Sl_clotho_bbcFsd180_250912/checkpoint-10260 \
    --model_base /mnt/sh/mmvision/home/changlitang/models/Qwen2.5-Omni-7B \
    --deepspeed 0 --use_beats --use_type_sampler \
    --use_lora --train_proj --train_classify --classify_type all_layer --gpu_num 2
