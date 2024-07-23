from abc import *
from typing import Optional
import torch
import os
import json
import torchvision.transforms.functional as TF

from synctweedies.method_configs.case_config import (CANONICAL_DENOISING_ZT,
                                                INSTANCE_DENOISING_XT,
                                                JOINT_DENOISING_XT,
                                                JOINT_DENOISING_ZT, METHOD_MAP)
from synctweedies.model.base_model import BaseModel
from synctweedies.utils.image_utils import merge_images
from synctweedies.utils.mesh_utils import split_groups


class WideImageModel(BaseModel):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{self.config.gpu}")
        super().__init__()
        self.initialize()
        
        
    def initialize(self):
        super().initialize()
        
        self.intermediate_dir = self.output_dir / "results"
        os.makedirs(self.intermediate_dir, exist_ok=True)
        
        log_opt = vars(self.config)
        config_path = os.path.join(self.output_dir, "wide_image_run_config.yaml")
        with open(config_path, "w") as f:
            json.dump(log_opt, f, indent=4)


    def get_views(self, panorama_height, panorama_width, window_size=None, stride=8):
        assert window_size != None
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


    def init_mapper(self, **kwargs):
        self.latent_canonical_height = self.config.panorama_height // 8
        self.latent_canonical_width = self.config.panorama_width // 8
        self.latent_instance_size = self.config.latent_instance_size  # only for SD
        self.rgb_instance_size = self.config.rgb_instance_size

        # Mapper guiding start / end of width / height in canonical space 
        self.mapper = self.get_views(self.latent_canonical_height, self.latent_canonical_width, window_size=self.latent_instance_size, stride=self.config.window_stride)
        self.count = torch.zeros(1, 4, self.latent_canonical_height, self.latent_canonical_width).to(self.device)
        self.value = torch.zeros(1, 4, self.latent_canonical_height, self.latent_canonical_width).to(self.device)
        self.num_views = len(self.mapper)

        self.rgb_mapper = self.get_views(self.config.panorama_height, self.config.panorama_width, window_size=self.rgb_instance_size, stride=self.config.window_stride * 8)
        self.rgb_count = torch.zeros(1, 3, self.config.panorama_height, self.config.panorama_width).to(self.device)
        self.rgb_value = torch.zeros(1, 3, self.config.panorama_height, self.config.panorama_width).to(self.device)


        # Cross attention mask
        self.attention_mask = []
        for i in range(self.num_views):  # N-1 0 1, 0 1 2, 1 2 3, ...
            self.attention_mask.append(
                [(i - 1 + self.num_views) % self.num_views, i, (i + 1) % self.num_views]
            )

        # Reference view
        ref_views = [self.num_views // 2]
        
        self.group_metas = split_groups(self.attention_mask, 1000, ref_views)
        
        
    def compute_noise_preds(
        self, 
        xts, 
        timestep, 
        **kwargs
    ):
     
        return self.model.compute_noise_preds(xts, timestep, **kwargs)


    def forward_mapping(self, z_t, **kwargs):
        """
        Return {x_t}^i from z_t
        Input:
            z_t: [1,C,H,W]
        Output:
            {x_t^i}: [N,C,h,w], where N denotes # of variables.
        """

        forwarded_instance_latents = []
        for h_start, h_end, w_start, w_end in self.mapper:
            forwarded_instance_latents.append(z_t[:, :, h_start:h_end, w_start:w_end])

        xts = torch.cat(forwarded_instance_latents, dim=0)

        return xts
        

    def inverse_mapping(self, x_ts, **kwargs):
        """
        Return z_t from {x_t}^i
        Input:
            x_ts: {x_t^i}: [N,C,h,w]
        Output:
            z_t: [1,C,H,W]
        """

        self.count.zero_()
        self.value.zero_()
        for i, (h_start, h_end, w_start, w_end) in enumerate(self.mapper):
            self.value[:, :, h_start:h_end, w_start:w_end] += x_ts[i:i+1, ...]
            self.count[:, :, h_start:h_end, w_start:w_end] += 1

        z_t = torch.where(self.count > 0, self.value / self.count, self.value)

        return z_t

    def init_prompt_embeddings(
        self, prompt: Optional[str] = None, negative_prompt: Optional[str] = None
    ):
        if negative_prompt is None:
            negative_prompt = self.config.negative_prompt
            
        self.config.prompt = prompt
        self.config.negative_prompt = negative_prompt
        self.prompt_embeds = self.compute_prompt_embeds(prompt, negative_prompt)

    def compute_prompt_embeds(
        self, prompt: Optional[str] = None, negative_prompt: Optional[str] = None
    ):
        if prompt is None:
            prompt = self.config.prompt
        if negative_prompt is None:
            negative_prompt = self.config.negative_prompt

        prompt_embeds = self.model._encode_prompt(
            prompt,
            do_classifier_free_guidance=True,
            num_images_per_prompt=1,
            negative_prompt=negative_prompt,
            device=self.device,
        )

        return prompt_embeds

    
    def initialize_latent(self, latent_space, **kwargs):
        model = kwargs.get("model")
        generator= kwargs.get("generator")
        device=self.device
        if latent_space == "instance":
            latent = model.prepare_latents(
                self.num_views, 
                4, 
                self.latent_instance_size * 8, 
                self.latent_instance_size * 8, 
                self.prompt_embeds.dtype, 
                device, 
                generator,
                None,)
            
        elif latent_space == "canonical":
            latent = model.prepare_latents(
                1, 
                4, 
                self.latent_canonical_height * 8, 
                self.latent_canonical_width * 8, 
                self.prompt_embeds.dtype, 
                device, 
                generator,
                None,)

        return latent

    @torch.no_grad()
    def __call__(self):
        self.config.prompt = f"Best quality, extremely detailed {self.config.prompt}"
        self.init_prompt_embeddings(self.config.prompt, self.config.negative_prompt)

        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        zts = self.initialize_latent(
            "canonical", generator=generator, model=self.model
        )
        xts = self.initialize_latent(
            "instance", generator=generator, model=self.model
        )

        case_name = METHOD_MAP[str(self.config.case_num)]
        if (self.config.case_num in INSTANCE_DENOISING_XT or self.config.case_num in JOINT_DENOISING_XT) and self.config.initialize_xt_from_zt:
            print(f"Initial z_T is given. x_ts are obtained from forward operation.")
            xts = self.forward_mapping(zts, bg=None)
        else:
            print(f"Initial z_T is NOT given. x_ts are individually initialized.")

        input_params = {
            "zts": zts, 
            "xts": xts
        }
        
        self.model.scheduler.set_timesteps(
            self.config.num_inference_steps, device=self.device
        )
        timesteps = self.model.scheduler.timesteps
        
        num_inference_steps = self.config.num_inference_steps
        num_timesteps = self.model.scheduler.config.num_train_timesteps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model.scheduler.order
        
        alphas = self.model.scheduler.alphas_cumprod ** (0.5)
        sigmas = (1 - self.model.scheduler.alphas_cumprod) ** (0.5)
        
        func_params = {
            "group_metas": self.group_metas,
            "ref_attention_end": self.config.ref_attention_end,
            "num_timesteps": num_timesteps,
            "prompt_embeds": self.prompt_embeds,
            "guidance_scale": self.config.guidance_scale,
        }
        with self.model.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                func_params["t"]  = t
                
                if t > (1-self.config.mvd_end) * num_timesteps:
                    out_params = self.one_step_process(
                        input_params,
                        t,
                        alphas,
                        sigmas,
                        case_name=case_name,
                        **func_params,
                    )
                else:
                    xts = input_params["xts"]
                    eps_preds = self.compute_noise_preds(xts, t, **func_params)
                    x0s = self.compute_tweedie(
                        xts, eps_preds, t, self.alphas, self.sigmas, **func_params
                    )
                    x_t_1 = self.compute_prev_state(xts, x0s, t, eps_preds, **func_params)
                    z_t_1 = self.inverse_mapping(x_t_1)

                    out_params = {
                        "x0s": x0s,
                        "z0s": None,
                        "x_t_1": x_t_1,
                        "z_t_1": z_t_1,
                    }

                assert out_params["x_t_1"] != None or out_params["z_t_1"] != None
                if out_params["x_t_1"] != None:
                    xts = out_params["x_t_1"]
                else:
                    xts = self.forward_mapping(out_params["z_t_1"], bg=None)
                
                if out_params["z_t_1"] != None:
                    zts = out_params["z_t_1"]
                else:
                    zts = self.inverse_mapping(out_params["x_t_1"])

                if out_params["x0s"] != None:
                    log_x0s = out_params["x0s"]
                else:
                    log_x0s = self.forward_mapping(out_params["z0s"], bg=None)

                log_x_prevs = xts
                input_params = {
                    "zts": zts, 
                    "xts": xts
                }

                """ Logging """
                if (i + 1) % self.config.log_step == 0:
                    log_x_prev_img = self.xs_to_pil_img(log_x_prevs)
                    log_x0_img = self.xs_to_pil_img(log_x0s)

                    log_img = merge_images([log_x_prev_img, log_x0_img])
                    log_img.save(f"{self.intermediate_dir}/i={i}_t={t}.png")

                    for view_idx in range(0, log_x0s.shape[0]):
                        if view_idx == 5:
                            break
                        log_x0 = log_x0s[view_idx]
                        decoded = self.decode_latents(log_x0.unsqueeze(0)).float()
                        TF.to_pil_image(decoded[0].cpu()).save(f"{self.intermediate_dir}/i={i}_v={view_idx}_view.png")
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model.scheduler.order == 0):
                    progress_bar.update()

        save_each_view_dir = self.output_dir / "results"
        os.makedirs(save_each_view_dir, exist_ok=True)
        for view_idx in range(0, xts.shape[0]):
            log_final_img = xts[view_idx]
            h_start, h_end, w_start, w_end = self.rgb_mapper[view_idx]
            decoded = self.decode_latents(log_final_img.unsqueeze(0)).float()
            save_path = save_each_view_dir / f"{view_idx}_wrange_{w_start}_{w_end}.png"
            TF.to_pil_image(decoded[0].cpu()).save(save_path)

        if (self.config.case_num in INSTANCE_DENOISING_XT or self.config.case_num in JOINT_DENOISING_XT):
            final_denoised = self.forward_mapping(
                self.inverse_mapping(out_params["x_t_1"])
            )
        elif (self.config.case_num in CANONICAL_DENOISING_ZT or self.config.case_num in JOINT_DENOISING_ZT):
            final_denoised = self.forward_mapping(out_params["z_t_1"])
        else:
            final_denoised = out_params["x_t_1"]

        final_denoised_img = self.xs_to_pil_img(final_denoised)
        final_denoised_img.save(f"{self.output_dir}/final_{self.config.seed}.png")


    @torch.no_grad()
    def decode_latents(self, latents):
        latents = latents.half()
        latents = 1 / 0.18215 * latents
        imgs = self.model.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def xs_to_pil_img(self, xs):
        """
        Input:
            x: [N,C,h,w]
        Otuput:
            panorama img
        """
        C, h, w = xs.shape[-3], xs.shape[-2], xs.shape[-1]
        xs = xs.reshape(-1, C, h, w)

        decoded_latents = []
        for view_idx in range(self.num_views):
            current_latent = xs[view_idx : view_idx + 1]
            decoded_latents.append(self.decode_latents(current_latent).float())
        
        decoded_latents = torch.cat(decoded_latents, dim=0)

        self.rgb_count.zero_()
        self.rgb_value.zero_()
        for i, (h_start, h_end, w_start, w_end) in enumerate(self.rgb_mapper):
            self.rgb_value[:, :, h_start:h_end, w_start:w_end] += decoded_latents[i:i+1, ...]
            self.rgb_count[:, :, h_start:h_end, w_start:w_end] += 1

        rgb_pano = torch.where(self.rgb_count > 0, self.rgb_value / self.rgb_count, self.rgb_value)

        return TF.to_pil_image(rgb_pano[0].cpu())