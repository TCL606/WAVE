#!/bin/bash

cd "$(cd $(dirname $0); pwd)"

#  --pred_embeds

bash test.sh \
    --interval 0.5 \
    --run_name debug4 \
    --dataset scripts/ret_v2a.json \
    --max_frames 128 \
    --model /mnt/sh/mmvision/home/changlitang/models/WAVE-7B-0211 \
    --model_base /mnt/sh/mmvision/home/changlitang/models/WAVE-7B-0211 \
    --use_beats --deepspeed 0 --train_classify --classify_type all_layer --pred_embeds --gpu_num 1

# bash test.sh \
#     --interval 0.5 \
#     --run_name debug3 \
#     --dataset scripts/ret_av2t.json \
#     --max_frames 128 \
#     --model /mnt/sh/mmvision/home/changlitang/models/WAVE-7B-0211 \
#     --model_base /mnt/sh/mmvision/home/changlitang/models/WAVE-7B-0211 \
#     --use_beats --deepspeed 0 --train_classify --classify_type all_layer --pred_embeds --gpu_num 1


# bash test.sh \
#     --interval 0.5 \
#     --run_name debug2 \
#     --dataset scripts/ret_mc.json \
#     --max_frames 128 \
#     --model /mnt/sh/mmvision/home/changlitang/models/WAVE-7B-0211 \
#     --model_base /mnt/sh/mmvision/home/changlitang/models/WAVE-7B-0211 \
#     --use_beats --deepspeed 0 --train_classify --classify_type all_layer --gpu_num 1