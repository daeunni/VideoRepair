import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
import warnings;warnings.filterwarnings('ignore')
from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling, batch_ddim_sampling_multidiffusion
from utils.utils import instantiate_from_config

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default='./checkpoints/base_512_v2/model.ckpt' , help="checkpoint path")
    parser.add_argument("--config", type=str, default = './configs/inference_t2v_512_v2.0.yaml', help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=2, help="batch size for inference")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=28)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")

    ## NOTE ours added 
    parser.add_argument("--mask_path", type=str, default = None, help="mask path")

    return parser


def run_inference(args, model, gpu_num, gpu_no, **kwargs):
    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"

    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels
    
    ## saving folders
    os.makedirs(args.savedir, exist_ok=True)

    ## step 2: load data
    ## -----------------------------------------------------------------
    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"

    with open(args.prompt_file, "r") as file:
        prompt_list = [line.strip() for line in file]       

    num_samples = len(prompt_list)
    filename_list = [f"{id+1:04d}" for id in range(num_samples)]

    samples_split = num_samples // gpu_num
    residual_tail = num_samples % gpu_num
    print(f'[rank:{gpu_no}] {samples_split}/{num_samples} samples loaded.')
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    if gpu_no == 0 and residual_tail != 0:
        indices = indices + list(range(num_samples-residual_tail, num_samples))
    prompt_list_rank = [prompt_list[i] for i in indices]

    ## conditional input
    if args.mode == "i2v":
        ## each video or frames dir per prompt
        cond_inputs = get_filelist(args.cond_input, ext='[mpj][pn][4gj]')   # '[mpj][pn][4gj]'
        assert len(cond_inputs) == num_samples, f"Error: conditional input ({len(cond_inputs)}) NOT match prompt ({num_samples})!"
        filename_list = [f"{os.path.split(cond_inputs[id])[-1][:-4]}" for id in range(num_samples)]
        cond_inputs_rank = [cond_inputs[i] for i in indices]

    filename_list_rank = [filename_list[i] for i in indices]

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    start = time.time()
    n_rounds = len(prompt_list_rank) // args.bs
    n_rounds = n_rounds+1 if len(prompt_list_rank) % args.bs != 0 else n_rounds

    for idx in range(0, n_rounds):
        print(f'[rank:{gpu_no}] batch-{idx+1} ({args.bs})x{args.n_samples} ...')
        idx_s = idx*args.bs
        idx_e = min(idx_s+args.bs, len(prompt_list_rank))
        batch_size = idx_e - idx_s
        filenames = filename_list_rank[idx_s:idx_e]

        noise_shape = [batch_size, channels, frames, h, w]   

        fps = torch.tensor([args.fps]*batch_size).to(model.device).long()

        prompts = prompt_list_rank[idx_s:idx_e]
        if isinstance(prompts, str):
            prompts = [prompts]

        #prompts = batch_size * [""]
        text_emb = model.get_learned_conditioning(prompts)       

        if args.mode == 'base':
            cond = {"c_crossattn": [text_emb], "fps": fps}   



        elif args.mode == 'i2v':
            #cond_images = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            cond_images = load_image_batch(cond_inputs_rank[idx_s:idx_e], (args.height, args.width))
            cond_images = cond_images.to(model.device)
            img_emb = model.get_image_embeds(cond_images)
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            cond = {"c_crossattn": [imtext_cond], "fps": fps}
        else:
            raise NotImplementedError

        for LOCAL_SEED in random.sample(range(1, 100000), 5) : 
            batch_samples = batch_ddim_sampling_multidiffusion(model, cond, \
                                                                noise_shape, args.n_samples, \
                                                                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, None, 
                                                                args.mask_path, LOCAL_SEED, args.seed, **kwargs)

            filenames = [prompt_list[0] + '_' + prompt_list[1] + '_' + str(LOCAL_SEED)]
            save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1

    start = time.time()

    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_num)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    print(f"Time used: {(time.time() - start):.2f} seconds")

    start = time.time()

    run_inference(args, model, gpu_num, rank)
    print(f"Inference Time used: {(time.time() - start):.2f} seconds")  