import torch
import os 
from diffusers import DiffusionPipeline
from PIL import Image 
import numpy as np 
from tqdm import tqdm 
import cv2
from typing import List, Optional, Union, Dict, Any, Union, Tuple
import matplotlib.pyplot as plt
from diffusers import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from lvdm.models.ddpm3d import LatentDiffusion
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from torchvision import transforms as tvt

logger = logging.get_logger(__name__)  


class Multidiffusion_T2VTurboVC2Pipeline(DiffusionPipeline):
    def __init__(
        self,
        pretrained_t2v: LatentDiffusion,
        scheduler: T2VTurboScheduler,
        model_config: Dict[str, Any] = None,
    ):
        super().__init__()

        self.register_modules(
            pretrained_t2v=pretrained_t2v,
            scheduler=scheduler,
        )
        self.vae = pretrained_t2v.first_stage_model
        self.unet = pretrained_t2v.model.diffusion_model
        self.text_encoder = pretrained_t2v.cond_stage_model     # lvdm.modules.encoders.condition.FrozenOpenCLIPEmbedder

        self.model_config = model_config
        self.vae_scale_factor = 8


    def match_noise_statistics(self, source_noise, target_noise):
        source_mean, source_var = source_noise.mean(), source_noise.var()
        target_mean, target_var = target_noise.mean(), target_noise.var()
        scaled_noise = (target_noise - target_mean) * (source_var / target_var).sqrt() + source_mean
        return scaled_noise

    def check_remain(self, changed, origin, x1, x2, y1, y2) : 
        unchanged_mask = torch.ones_like(changed, dtype=bool)
        unchanged_mask[:, :, :, y1:y2, x1:x2] = False
        unchanged_latents = changed[unchanged_mask]
        original_latents = origin[unchanged_mask]
        are_identical = torch.equal(unchanged_latents, original_latents)
        return are_identical

    def load_image(self, imgname, target_size, b_width, b_height) :
        pil_img = Image.open(imgname).convert('RGB').resize((b_width, b_height))
        if target_size is not None:
            if isinstance(target_size, int):
                target_size = (target_size, target_size)
            pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension
    

    def img_to_latents(self, x: torch.Tensor, vae: AutoencoderKL):
        x = 2. * x - 1.
        posterior = vae.encode(x).latent_dist
        latents = posterior.mean * 0.18215
        return latents


    def get_views(self, panorama_height, panorama_width, window_size=64, stride=8):
        panorama_height /= 8      # 40 
        panorama_width /= 8       # 64 

        num_blocks_height = (panorama_height - window_size) // stride + 1
        num_blocks_width = (panorama_width - window_size) // stride + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))
        return views


    def preprocess_mask(self, mask_path, h, w, device):
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        mask = torch.from_numpy(mask).to(device)
        mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
        return mask

    @torch.no_grad()
    def encode_imgs(self, imgs):  
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def get_random_background(self, n_samples):
        # sample random background with a constant rgb value
        backgrounds = torch.rand(n_samples, 3, device=self.device)[:, :, None, None].repeat(1, 1, 512, 512)
        return torch.cat([self.encode_imgs(bg.unsqueeze(0)) for bg in backgrounds])


    @torch.no_grad()
    def ddim_inversion(self, imgname, num_steps, verify, save_path, b_width, b_height):

        print('DDIM inversion start!')
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16

        inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='scheduler')
        pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1',
                                                    scheduler=inverse_scheduler,
                                                    safety_checker=None,
                                                    torch_dtype=dtype)
        pipe.to(device)
        vae = pipe.vae

        input_img = self.load_image(imgname, None, b_width, b_height).to(device=device, dtype=dtype)
        latents = self.img_to_latents(input_img, vae)

        inv_latents, _ = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                            width=input_img.shape[-1], height=input_img.shape[-2],
                            output_type='latent', return_dict=False,
                            num_inference_steps=num_steps, latents=latents) 

        # latent save 
        torch.save(inv_latents, save_path + '/ddim_inversion.pt')

        # check replacement 
        if verify:
            pipe.scheduler = DDIMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='scheduler')
            image = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                        num_inference_steps=num_steps, latents=inv_latents)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(tvt.ToPILImage()(input_img[0]))
            ax[1].imshow(image.images[0])
            plt.savefig(save_path + '/inversion_check.png')
            plt.show()
        return inv_latents



    ## Prompt embedding 
    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        prompt_embeds: None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        if prompt_embeds is None:
            prompt_embeds = self.text_encoder(prompt)

        prompt_embeds = prompt_embeds.to(device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape        # [1, 77, 1024]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)    # num_videos_per_prompt = 1 
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_videos_per_prompt, seq_len, -1        # [1, 77, 1024]
        )

        # Don't need to get uncond prompt embedding because of LCM Guided Distillation
        return prompt_embeds

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        frames,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings
        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb


    @torch.no_grad()    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,             # origin prompt 
        all_prompt: Union[str, List[str]] = None,       # local prompt 
        mask_path: Union[str, List[str]] = None,          # binary mask path 
        round_num: int = None,  
        noise_map: Union[str, List[str]] = None,  
        height: Optional[int] = 320,
        width: Optional[int] = 512,
        frames: int = 16,
        fps: int = 16,
        guidance_scale: float = 7.5,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 4,
        lcm_origin_steps: int = 50,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        origin_seed: int = 123, 
        local_seed: int = None, 
        ):
        unet_config = self.model_config["params"]["unet_config"]    

        # 0. Default height and width to unet
        frames = self.pretrained_t2v.temporal_length if frames < 0 else frames

        # 2. Define call parameters
        if all_prompt is not None and isinstance(all_prompt, str):
            batch_size = 1
        elif all_prompt is not None and isinstance(all_prompt, list):
            batch_size = len(all_prompt)         # len == fg+1
        else:
            batch_size = prompt_embeds.shape[0]
        print('BS : ', batch_size)

        device = self._execution_device
        
        fg_masks = torch.cat([self.preprocess_mask(m_path, height // 8 , width // 8, device) for m_path in mask_path])       # [fg, 1, 40, 64]
        bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)           # [1, 1, 40, 64]
        bg_mask[bg_mask < 0] = 0

        bg_mask = bg_mask.unsqueeze(2).repeat(1, 1, 16, 1, 1)           # [1, 1, 16, 40, 64]
        fg_masks = fg_masks.unsqueeze(2).repeat(1, 1, 16, 1, 1)         # [fg, 16, 40, 64]
        masks = torch.cat([bg_mask, fg_masks])                          # [1+fg, 1, 16, 40, 64] -> mask concat 

        # 3. Encode input prompt (Original)
        bg_prompt_embeds = self._encode_prompt(       # [1, 77, 1024]
            all_prompt[0],        # background prompt (remain)
            device,
            num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
        )
        fg_prompt_embeds1 = self._encode_prompt(       # [1, 77, 1024]
            all_prompt[1],       # local prompt 
            device, 
            num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
        )
        prompt_embeds = torch.cat((bg_prompt_embeds, fg_prompt_embeds1), dim=0)     

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, lcm_origin_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variable
        num_channels_latents = unet_config["params"]["in_channels"]

        latents = self.prepare_latents(      
            1, 
            num_channels_latents,     # unet in-channel
            frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )                # [1, 4, 16, 40, 64]

        bs = batch_size * num_videos_per_prompt

        # 6. Get Guidance Scale Embedding
        w = torch.tensor(guidance_scale).repeat(bs)
        w_embedding = self.get_w_embedding(w, embedding_dim=256).to(device)       

        len_prompts = len(all_prompt)   

        count = torch.zeros_like(latents).to(device)     
        value = torch.zeros_like(latents).to(device) 
        value2 = torch.zeros_like(latents).to(device) 

        # When multidiffusion denoising 
        with self.progress_bar(total=num_inference_steps) as progress_bar:

            for i, t in enumerate(timesteps):   
                count.zero_()    
                value.zero_()   
                value2.zero_()
                ts = torch.full((bs,), t, device=device, dtype=torch.long)  
                context = {"context": torch.cat([prompt_embeds.float()], 1), "fps": fps}   # [fg+1, 77, 1024]
                masks_view = masks   # [fg+1, 1, 16, 40, 40]

                if i == 0 :    

                    if (round_num == 0) or (not os.path.exists(noise_map)) : 
                        original_latent_view = latents                  
                    
                    else : 
                        original_latent_view = torch.load(noise_map)

                    torch.manual_seed(local_seed)
                    new_rerandom = torch.randn_like(latents, device=device)       
                    torch.manual_seed(origin_seed)

                    if torch.equal(original_latent_view, new_rerandom) : 
                        print('Tensors are same')
      
                    latents = torch.where(bg_mask == 1, original_latent_view, new_rerandom)     

                    name = f'init_latent_{local_seed}.pt'
                    latent_path = mask_path[0].replace('binary_yes_mask.png', name)
                    torch.save(latents, latent_path)
                
                latent_view = torch.squeeze(latents).repeat(len_prompts, 1, 1, 1, 1)        

                noise_pred = self.unet(latent_view, 
                                        ts,                  
                                        **context, timestep_cond=w_embedding.to(self.dtype),)  

                # compute the previous noisy sample x_t -> x_t-1 
                denoised_latents_view, denoised = self.scheduler.step(noise_pred, i, t, latent_view, return_dict=False) 

                value += (denoised_latents_view * masks_view).sum(dim=0, keepdims=True)      
                value2 += (denoised * masks_view).sum(dim=0, keepdims=True)           
                count += masks_view.sum(dim=0, keepdims=True)         
                latents = torch.where(count > 0, value / count, value)    
                denoised = torch.where(count > 0, value2 / count, value2) 

                progress_bar.update()


        if not output_type == "latent":
            videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)         
        else:
            videos = denoised

        return videos

