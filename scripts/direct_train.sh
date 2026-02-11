#!/bin/bash

cd "$(cd $(dirname $0); pwd)"

data=scripts/ret_av2t.json

bash train.sh \
    --interval 0.5 \
    --run_name debug \
    --dataset $data \
    --lr 2e-5 --bs 1 --epoch 5 \
    --max_frames 128 --save_steps 50000 \
    --model /mnt/sh/mmvision/home/changlitang/models/WAVE-7B \
    --model_base /mnt/sh/mmvision/home/changlitang/models/WAVE-7B \
    --deepspeed 0 --use_beats --use_type_sampler \
    --use_lora --train_proj --train_classify --classify_type all_layer --gpu_num 2
