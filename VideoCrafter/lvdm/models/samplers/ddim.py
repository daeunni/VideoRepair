import numpy as np
from tqdm import tqdm
import torch
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from lvdm.common import noise_like
from PIL import Image


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
        self.use_scale = self.model.use_scale
        print('DDIM scale', self.use_scale)

        if self.use_scale:
            self.register_buffer('scale_arr', to_torch(self.model.scale_arr))
            ddim_scale_arr = self.scale_arr.cpu()[self.ddim_timesteps]
            self.register_buffer('ddim_scale_arr', ddim_scale_arr)
            ddim_scale_arr = np.asarray([self.scale_arr.cpu()[0]] + self.scale_arr.cpu()[self.ddim_timesteps[:-1]].tolist())
            self.register_buffer('ddim_scale_arr_prev', ddim_scale_arr)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # 이 함수는 공용으로 사용할까? 
    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               multidiffusion=None,       # NOTE
               mask_path = None,          # NOTE
               local_seed = None,         # NOTE
               global_seed = None, 
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=schedule_verbose)
        
        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        # 여기서 들어가는 배치를 조절해야하나? 
        
        # sampling!! 
        if multidiffusion : 
            samples, intermediates = self.ddim_sampling(conditioning, size,
                                            mask_path = mask_path,          # NOTE 
                                            local_seed = local_seed,        # NOTE 
                                            global_seed = global_seed, 
                                            callback=callback,
                                            img_callback=img_callback,
                                            quantize_denoised=quantize_x0,
                                            mask=mask, x0=x0,
                                            ddim_use_original_steps=False,
                                            noise_dropout=noise_dropout,
                                            temperature=temperature,
                                            score_corrector=score_corrector,
                                            corrector_kwargs=corrector_kwargs,
                                            x_T=x_T,
                                            log_every_t=log_every_t,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=unconditional_conditioning,           # text embedding이 들어감 
                                            verbose=verbose,
                                            **kwargs)
        else : 
            samples, intermediates = self.ddim_sampling_origin(conditioning, size,
                                                        callback=callback,
                                                        img_callback=img_callback,
                                                        quantize_denoised=quantize_x0,
                                                        mask=mask, x0=x0,
                                                        ddim_use_original_steps=False,
                                                        noise_dropout=noise_dropout,
                                                        temperature=temperature,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs,
                                                        x_T=x_T,
                                                        log_every_t=log_every_t,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,           # text embedding이 들어감 
                                                        verbose=verbose,
                                                        **kwargs)

        return samples, intermediates        # -> step별 결과가 적재되어 있는 noise list들임 
        

    def preprocess_mask(self, mask_path, h, w, device):

        # import pdb;pdb.set_trace()

        mask = np.array(Image.open(mask_path).convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        mask = torch.from_numpy(mask).to(device)
        mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
        return mask

    # multidiffusion 모드일 때 
    @torch.no_grad()
    def ddim_sampling(self, cond, shape, mask_path=None, local_seed=None, global_seed=None, 
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,
                      cond_tau=1., target_size=None, start_timesteps=None,
                      **kwargs):

        device = self.model.betas.device        
        print('ddim device', device)
        print('Local seed: ', local_seed)

        # NOTE mask load 
        height = 320 ; width = 512
        fg_masks = torch.cat([self.preprocess_mask(mask_path, height // 8 , width // 8, device)]) 
        bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)     
        bg_mask[bg_mask < 0] = 0
        bg_mask = bg_mask.unsqueeze(2).repeat(1, 1, 16, 1, 1) 
        fg_masks = fg_masks.unsqueeze(2).repeat(1, 1, 16, 1, 1) 
        masks_view = torch.cat([bg_mask, fg_masks])    

        # latent image 
        b = shape[0]       # 배치가 여기로 들어온다 (1)
        if x_T is None:     
            img = torch.randn(shape, device=device)   # Noise input (bs, 4, 16, 40, 64) : shape -> 랜덤 initialize? ㅇㅇ
        else:
            img = x_T

        # latent와 같은 역할임! 
        img = img[0, :, :, :, :].unsqueeze(dim=0)

        count = torch.zeros_like(img).to(device)     # torch.zeros_like(latents)
        value = torch.zeros_like(img).to(device) 
        value2 = torch.zeros_like(img).to(device)

        # NOTE (7/30) Re-randomize  
        # rerandom = False 
        # if rerandom: 
        #     print('Gonna re-randomize w/ local bounding box')
        #     img_copy = img.clone()
        #     # bounding box 
        #     y1, y2 = 25, 35  
        #     x1, x2 = 14, 19 
        #     new_noise = torch.randn((b, shape[1], shape[2], y2 - y1, x2 - x1), device=device)    # [1, 4, 16, 10, 15]
        #     img[:, :, :, y1:y2, x1:x2] = new_noise       # replace 
        #     are_equal = torch.equal(img_copy, img)
        #     assert are_equal == False
        
        # step justify
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        init_x0 = False
        clean_cond = kwargs.pop("clean_cond", False)

        ## step iteration -> 50 step 반복 
        for i, step in tqdm(enumerate(iterator)):
            count.zero_()    # [1, 4, 16, 40, 64]
            value.zero_()    # [1, 4, 16, 40, 64]
            value2.zero_()

            if i == 0 : 
                print('Re-randomize in the 1st latent noise')     
                original_latent_view = img 
                torch.manual_seed(local_seed)
                new_rerandom = torch.randn_like(img, device=device)
                torch.manual_seed(global_seed)

                if torch.equal(original_latent_view, new_rerandom) : 
                    print('Tensors are same')

                latent_view = torch.cat((original_latent_view, new_rerandom), dim=0)     # 배치 2로 만들기! 
            
            else : 
                latent_view = torch.squeeze(img).repeat(b, 1, 1, 1, 1) 

            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # 우리 : start_timesteps == None 
            if start_timesteps is not None:
                assert x0 is not None
                if step > start_timesteps*time_range[0]:
                    continue
                elif not init_x0:
                    img = self.model.q_sample(x0, ts) 
                    init_x0 = True

            # use mask to blend noised original latent (img_orig) & new sampled latent (img)
            # 우리 : mask == None 
            if mask is not None:
                assert x0 is not None
                if clean_cond:
                    img_orig = x0
                else:
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
                img = img_orig * mask + (1. - mask) * img # keep original & modify use img
            
            index_clip =  int((1 - cond_tau) * total_steps)

            # 우리 : target_size == None 
            if (index <= index_clip) and (target_size is not None):
                target_size_ = [target_size[0], target_size[1]//8, target_size[2]//8]
                img = torch.nn.functional.interpolate(
                img,
                size=target_size_,
                mode="nearest",
                )


            # unet + scheduling (p_sample_ddim 이건 안건들어도 될 듯)
            # 배치가 2라고 생각하고 각 프로세스에 서로 다른 prompt를 줘야할 듯? 
            img_view, pred_x0 = self.p_sample_ddim(latent_view, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,          # test embeddinng이 들어감 ['c_crossattn', 'fps']
                                      x0=x0,
                                      **kwargs)

                                      # unconditional_conditioning['c_crossattn'][0].shape = [1, 77, 1024]
            # denoised_latents_view, denoised

            # NOTE multidiffusion 
            value += (img_view * masks_view).sum(dim=0, keepdims=True) 
            value2 += (pred_x0 * masks_view).sum(dim=0, keepdims=True)    
            count += masks_view.sum(dim=0, keepdims=True) 

            img = torch.where(count > 0, value / count, value)  
            pred_x0 = torch.where(count > 0, value2 / count, value2) 

            # img, pred_x0 = outs

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            # 아 log_every_t 마다 logging을 해서 약간 진짜 intermediate 같은거구낭 
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates     # t2v turbo에서 denoised_latents_view, denoised 

        # img = 1, 4, 16, 40, 64]
        # intermediates = ['x_inter', 'pred_x0']

    # 여기는 안건들어도 될 듯? 
    @torch.no_grad()
    def p_sample_ddim(self, x,      # x == image 
                      c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      uc_type=None, conditional_guidance_scale_temporal=None, **kwargs):

        b, *_, device = *x.shape, x.device

        if x.dim() == 5:
            is_video = True
        else:
            is_video = False
        
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, **kwargs) # unet denoiser

        else:
            # with unconditional condition -> text condition이 있을 때? 
            if isinstance(c, torch.Tensor):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)

            elif isinstance(c, dict):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)

            else:
                raise NotImplementedError

            # text cfg
            if uc_type is None:
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            else:
                if uc_type == 'cfg_original':
                    e_t = e_t + unconditional_guidance_scale * (e_t - e_t_uncond)
                elif uc_type == 'cfg_ours':
                    e_t = e_t + unconditional_guidance_scale * (e_t_uncond - e_t)
                else:
                    raise NotImplementedError

            # temporal guidance
            if conditional_guidance_scale_temporal is not None:
                e_t_temporal = self.model.apply_model(x, t, c, **kwargs)
                e_t_image = self.model.apply_model(x, t, c, no_temporal_attn=True, **kwargs)
                e_t = e_t + conditional_guidance_scale_temporal * (e_t_temporal - e_t_image)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)

        # 스케줄링에 필요한 것들이 아닐지.. 
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)


        # direction pointing to x_t -> 이게 스케줄링 같네 
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        if self.use_scale:
            scale_arr = self.model.scale_arr if use_original_steps else self.ddim_scale_arr
            scale_t = torch.full(size, scale_arr[index], device=device)
            scale_arr_prev = self.model.scale_arr_prev if use_original_steps else self.ddim_scale_arr_prev
            scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
            pred_x0 /= scale_t 
            x_prev = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
        else:
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0


    # 원래 버전 (Multidiffusion 버전이랑 ddim process 따로 만들기)
    @torch.no_grad()
    def ddim_sampling_origin(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,
                      cond_tau=1., target_size=None, start_timesteps=None,
                      **kwargs):

        device = self.model.betas.device        
        print('ddim device', device)
        b = shape[0]       # 배치가 여기로 들어온다 (1)

        # latent image 
        if x_T is None:     
            img = torch.randn(shape, device=device)   # Noise input (1, 4, 16, 40, 64) : shape -> 랜덤 initialize? ㅇㅇ
        else:
            img = x_T

        # step justify
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        init_x0 = False
        clean_cond = kwargs.pop("clean_cond", False)

        ## step iteration -> 50 step 반복 
        for i, step in enumerate(iterator):

            # NOTE 일단 여기서 mask를 먹을 수 있게 해야겠네 
            # NOTE 디퓨전 forward process도 두개 만들고 


            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # 우리 : start_timesteps == None 
            if start_timesteps is not None:
                assert x0 is not None
                if step > start_timesteps*time_range[0]:
                    continue
                elif not init_x0:
                    img = self.model.q_sample(x0, ts) 
                    init_x0 = True

            # use mask to blend noised original latent (img_orig) & new sampled latent (img)
            # 우리 : mask == None 
            if mask is not None:
                assert x0 is not None
                if clean_cond:
                    img_orig = x0
                else:
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
                img = img_orig * mask + (1. - mask) * img # keep original & modify use img
            
            index_clip =  int((1 - cond_tau) * total_steps)

            # 우리 : target_size == None 
            if (index <= index_clip) and (target_size is not None):
                target_size_ = [target_size[0], target_size[1]//8, target_size[2]//8]
                img = torch.nn.functional.interpolate(
                img,
                size=target_size_,
                mode="nearest",
                )


            # 이게 뭔 코드냐 -> unet + scheduling (p_sample_ddim 이건 안건들어도 될 듯)
            img, pred_x0 = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,          # test embeddinng이 들어감 ['c_crossattn', 'fps']
                                      x0=x0,
                                      **kwargs)

                                      # unconditional_conditioning['c_crossattn'][0].shape = [1, 77, 1024]
            # denoised_latents_view, denoised

            # img, pred_x0 = outs

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            # 아 log_every_t 마다 logging을 해서 약간 진짜 intermediate 같은거구낭 
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        # import pdb;pdb.set_trace()

        return img, intermediates     # t2v turbo에서 denoised_latents_view, denoised 

        # img = 1, 4, 16, 40, 64]
        # intermediates = ['x_inter', 'pred_x0'] -> 


    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)

        def extract_into_tensor(a, t, x_shape):
            b, *_ = t.shape
            out = a.gather(-1, t)
            return out.reshape(b, *((1,) * (len(x_shape) - 1)))

        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec

