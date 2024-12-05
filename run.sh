#!/bin/bash
output_root="./results/test/"
eval_sections=("count")   #  "color" "action"

for section in "${eval_sections[@]}"
do
    CUDA_VISIBLE_DEVICES=5,6 python main.py \
                        --output_root="$output_root" \
                        --eval_section="$section" \
                        --model='t2vturbo' \
                        --selection_score='dsg_blip' \
                        --seed=123 \
                        --round=1 \
                        --k=5 
done


