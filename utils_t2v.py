import os, subprocess, time 
import torch, torchvision
from concurrent.futures import ThreadPoolExecutor
from lightning_fabric import seed_everything

# For T2V-turbo 
from omegaconf import OmegaConf
from t2vturbo.utils.lora import collapse_lora, monkeypatch_remove_lora
from t2vturbo.utils.lora_handler import LoraHandler
from t2vturbo.utils.common_utils import load_model_checkpoint
from t2vturbo.utils.utils import instantiate_from_config
from t2vturbo.scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from t2vturbo.pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline
from t2vturbo.pipeline.multidiffusion import *        

def save_video(
    vid_tensor, metadata: dict, root_path="./", fps=16
    ):
    unique_name = metadata['prompt'].replace('\n', '') + '_' + str(metadata['idx']) + '.mp4'      
    print('unique_name :', unique_name)
    unique_name = os.path.join(root_path, unique_name)     
    print('Save path : ', unique_name)

    video = vid_tensor.detach().cpu()
    video = torch.clamp(video.float(), -1.0, 1.0)
    video = video.permute(1, 0, 2, 3)  # t,c,h,w
    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)

    torchvision.io.write_video(
        unique_name, video, fps=fps, video_codec="h264", options={"crf": "10"}
    )
    return unique_name

def save_videos(
    video_array, metadata: dict, fps: int = 16
):
    paths = []
    root_path = metadata["save_path"]        
    os.makedirs(root_path, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        paths = list(
            executor.map(
                save_video,
                video_array,
                [metadata] * len(video_array),
                [root_path] * len(video_array),
                [fps] * len(video_array),
            )
        )
    return paths[0]

def load_t2vturbo(SEED) : 
    torch.manual_seed(SEED)
    seed_everything(SEED)  

    start_time = time.time()
    unet_dir         = './checkpoints/t2v-turbo/unet_lora.pt'
    videocrafter_dir = './checkpoints/VideoCrafter/base_512_v2/model.ckpt'

    config = OmegaConf.load("./checkpoints/t2v-turbo/inference_t2v_512_v2.0.yaml")    # change to your path
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(pretrained_t2v, videocrafter_dir)

    unet_config = model_config["params"]["unet_config"]
    unet_config["params"]["time_cond_proj_dim"] = 256
    unet = instantiate_from_config(unet_config)
    unet.load_state_dict(
        pretrained_t2v.model.diffusion_model.state_dict(), strict=False
    )
    use_unet_lora = True
    lora_manager = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=use_unet_lora,
        save_for_webui=True,
        unet_replace_modules=["UNetModel"],
    )
    lora_manager.add_lora_to_model(
        use_unet_lora,
        unet,
        lora_manager.unet_replace_modules,
        lora_path=unet_dir,
        dropout=0.1,
        r=64,
    )
    unet.eval()
    collapse_lora(unet, lora_manager.unet_replace_modules)
    monkeypatch_remove_lora(unet)

    pretrained_t2v.model.diffusion_model = unet
    scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )
    # initial model pipeline load 
    pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config).to("cuda")
    multidiffusion_pipeline = Multidiffusion_T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config).to("cuda")   # NOTE multidiffusion pipeline is added  

    torch.manual_seed(SEED)
    seed_everything(SEED)      
    print('T2V turbo model loading time (min) : ', (time.time() - start_time)//60)

    return pipeline, multidiffusion_pipeline


def T2VTurbo_from_each_prompt(outpath,          # generation path 
                            prompt_text,       # count or action? 
                            seeds,
                            pipeline, 
                            ):  

        os.makedirs(outpath, exist_ok=True)
        res_dir = outpath

        torch.manual_seed(seeds)
        seed_everything(seeds)    

        result = pipeline(
            prompt=prompt_text,
            frames=16,
            fps=16,
            guidance_scale=7.5,
            num_inference_steps=4,
            num_videos_per_prompt=1,
        )
        paths = save_videos(
            result,              
            metadata={
                "prompt": prompt_text,
                "seed": seeds,
                "guidance_scale": 7.5,
                "num_inference_steps": 4,
                "save_path" : res_dir, 
                "idx" : 0,   
            },
            fps=16,
        )

def T2VTurbo_refinement(outpath,           
                        round_num, 
                        noise_map, 
                        prompt_text,        
                        seeds,
                        pipeline, 
                        all_prompt,      
                        local_seed, 
                        mask_path,         
                        suffix, 
                        ):  
        os.makedirs(outpath, exist_ok=True)
        res_dir = outpath

        torch.manual_seed(seeds)
        seed_everything(seeds)    

        print('Re-randomization seed: ', local_seed)

        result = pipeline(
            prompt=prompt_text,
            all_prompt=all_prompt,  
            mask_path=mask_path, 
            round_num = round_num, 
            noise_map = noise_map, 
            frames=16,
            fps=16,
            guidance_scale=7.5,
            num_inference_steps=4,
            num_videos_per_prompt=1,
            origin_seed = seeds, 
            local_seed = local_seed, 
        )
        paths = save_videos(
            result,              
            metadata={
                "prompt": prompt_text,
                "seed": seeds,
                "guidance_scale": 7.5,
                "num_inference_steps": 4,
                "save_path" : res_dir, 
                "idx" : suffix,   
            },
            fps=16,
        )

def VideoCrafter_from_each_prompt(
                            outpath,        
                            prompts_path, 
                            seeds = 123):  

        os.makedirs(outpath, exist_ok=True)

        # Base parameters for VideoCrafter2
        ckpt = './checkpoints/VideoCrafter/base_512_v2/model.ckpt'      # fix 
        config = './VideoCrafter/configs/inference_t2v_512_v2.0.yaml'    # fix 
        SEED = str(seeds)
        prompt_file = prompts_path


        # aruments 
        command = [
            "python3", "./VideoCrafter/scripts/evaluation/inference.py",
            "--seed", SEED,
            "--mode", "base",
            "--ckpt_path", ckpt,
            "--config", config,
            "--savedir", outpath,
            "--n_samples", "1",
            "--bs", "1",
            "--height", "320",
            "--width", "512",
            "--unconditional_guidance_scale", "12.0",
            "--ddim_steps", "50",
            "--ddim_eta", "1.0",
            "--prompt_file", prompt_file,
            "--fps", "28",
        ]
        print(command)

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print("Command executed successfully with output:")
            print(result.stdout)
        else:
            print("Command execution failed with error:")
            print(result.stderr)


def VideoCrafter_refinement(
                            outpath,          # generation path 
                            all_prompt,
                            mask_path, 
                            seeds):  

        os.makedirs(outpath, exist_ok=True)

        # Base parameters for VideoCrafter2
        ckpt = './checkpoints/VideoCrafter/base_512_v2/model.ckpt'       # fix 
        config = './VideoCrafter/configs/inference_t2v_512_v2.0.yaml'    # fix 
        SEED = str(seeds)

        # aruments 
        command = [
            "python3", "./VideoCrafter/scripts/evaluation/inference_multidiffusion.py",
            "--seed", SEED,
            "--mode", "base",
            "--ckpt_path", ckpt,
            "--config", config,
            "--savedir", outpath,
            "--n_samples", "1",
            "--bs", "2",
            "--height", "320",
            "--width", "512",
            "--unconditional_guidance_scale", "12.0",
            "--ddim_steps", "50",
            "--ddim_eta", "1.0",
            "--prompt_file", all_prompt,        # all prompt path 
            "--fps", "28", 
            "--mask_path", mask_path,            
        ]
        print(command)

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print("Command executed successfully with output:")
            print(result.stdout)
        else:
            print("Command execution failed with error:")
            print(result.stderr)