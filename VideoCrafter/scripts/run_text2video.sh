# bash scripts/run_text2video.sh
name="base_512_v2"

ckpt='checkpoints/base_512_v2/model.ckpt'
config='/nas-ssd2/daeun/OPT2I/VideoCrafter/configs/inference_t2v_512_v2.0.yaml'
prompt_file='/nas-ssd2/daeun/OPT2I/t2v-turbo/poc_prompts.txt'
res_dir='poc_results'

CUDA_VISIBLE_DEVICES=3 python3 scripts/evaluation/inference.py \
--seed 1214 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 28
