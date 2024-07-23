import json
import os
from abc import *

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import ControlNetModel
from diffusers.utils import is_compiled_module
from PIL import Image

from synctweedies.method_configs.case_config import * 
from synctweedies.model.base_model import BaseModel
from synctweedies.renderer.mesh.voronoi import voronoi_solve
from synctweedies.renderer.panorama.Equirec2Perspec import Equirectangular
from synctweedies.renderer.panorama.utils import *
from synctweedies.utils.image_utils import *
from synctweedies.utils.mesh_utils import *


class PanoramaModel(BaseModel):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{self.config.gpu}")
        super().__init__()
        self.initialize()
    
    
    def initialize(self):
        super().initialize()


    def init_mapper(
        self,
        canonical_rgb_h=None,
        canonical_rgb_w=None,
        canonical_latent_h=None,
        canonical_latent_w=None,
        instance_latent_size=None,
        instance_rgb_size=None,
        theta_range=None,
        theta_interval=None,
        phi_range=None,
        phi_interval=None,
    ):
        
        self.attention_mask = []
        if canonical_rgb_h is None:
            canonical_rgb_h = self.config.canonical_rgb_h
        if canonical_rgb_w is None:
            canonical_rgb_w = self.config.canonical_rgb_w
        if canonical_latent_h is None:
            canonical_latent_h = self.config.canonical_latent_h
        if canonical_latent_w is None:
            canonical_latent_w = self.config.canonical_latent_w
        if instance_latent_size is None:
            instance_latent_size = self.config.instance_latent_size
        if instance_rgb_size is None:
            instance_rgb_size = self.config.instance_rgb_size

        if theta_range is None:
            theta_range = self.config.theta_range
        if theta_interval is None:
            theta_interval = self.config.theta_interval
        if phi_range is None:
            phi_range = self.config.phi_range
        if phi_interval is None:
            phi_interval = self.config.phi_interval

        self.theta_list = list(range(theta_range[0], theta_range[1], theta_interval))
        self.phi_list = list(range(phi_range[0], phi_range[1], phi_interval))
        num_theta = len(self.theta_list)
        num_phi = len(self.phi_list)
        num_views = num_theta * num_phi
        self.num_views = num_views

        self.latent_xy_map_stack = self.compute_xy_map(
            canonical_latent_h, canonical_latent_w, instance_latent_size
        )
        self.latent_mask_stack = self.compute_mask_map(
            canonical_latent_h, canonical_latent_w, instance_latent_size
        )

        self.rgb_xy_map_stack = self.compute_xy_map(
            canonical_rgb_h, canonical_rgb_w, instance_rgb_size
        )
        self.rgb_mask_stack = self.compute_mask_map(
            canonical_rgb_h, canonical_rgb_w, instance_rgb_size
        )

        """ cross attention mask """
        self.attention_mask = []
        for i in range(self.num_views):  # N-1 0 1, 0 1 2, 1 2 3, ...
            self.attention_mask.append(
                [(i - 1 + self.num_views) % self.num_views, i, (i + 1) % self.num_views]
            )

        ref_views = [self.num_views // 2]
        self.group_metas = split_groups(self.attention_mask, 48, ref_views)
        

    def compute_xy_map(
        self, 
        canonical_h, 
        canonical_w, 
        instance_size,
    ):
        
        xy_map_stack = torch.zeros(
            (self.num_views, instance_size, instance_size, 2), device=self.device
        )
        idx = 0
        for theta in self.theta_list:
            for phi in self.phi_list:
                xy_map = compute_xy_map_per_angle(
                    self.config.FOV,
                    theta,
                    phi,
                    instance_size,
                    instance_size,
                    canonical_h,
                    canonical_w,
                    device=self.device,
                )
                xy_map_stack[idx] = xy_map
                idx += 1

        return xy_map_stack

    def compute_mask_map(
        self, 
        canonical_h, 
        canonical_w, 
        instance_size
    ):
    
        mask_map_stack = torch.zeros(
            (self.num_views, canonical_h, canonical_w), device=self.device
        )
        idx = 0
        for theta in self.theta_list:
            for phi in self.phi_list:
                xy_map = compute_mask_torch(
                    self.config.FOV,
                    theta,
                    phi,
                    instance_size,
                    instance_size,
                    canonical_h,
                    canonical_w,
                    device=self.device,
                )
                mask_map_stack[idx] = xy_map  # [H, W]
                idx += 1

        return mask_map_stack  # [num_views, canonical_h, canonical_w]
        

    def forward_ft(
        self, 
        canonical_input, 
        index, 
        xy_map_stack=None, 
        **kwargs,
    ):
        
        xy = xy_map_stack[index].to(canonical_input)
        instance_out = remap_torch(canonical_input, xy[..., 0], xy[..., 1])
        return instance_out
    

    def inverse_ft(
        self, 
        screen_input, 
        index, 
        orig_canonical_input, 
        xy_map_stack=None, 
        **kwargs,
    ):
        xy = xy_map_stack[index]
        pano_height, pano_width = (
            orig_canonical_input.shape[-2],
            orig_canonical_input.shape[-1],
        )
        canonical_out, mask = wrapper_perspective_to_pano_torch(
            screen_input, xy, pano_height, pano_width
        )
        return canonical_out, mask

    def forward_mapping(
        self, 
        z_t, 
        xy_map_stack=None, 
        encoding=False, 
        **kwargs,
    ):
        
        x_ts = [
            self.forward_ft(z_t, i, xy_map_stack, **kwargs)
            for i in range(self.num_views)
        ]
        x_ts = torch.cat(x_ts, 0)
        if encoding:
            encoded_x_ts = []
            for i in range(self.num_views):
                encoded = self.encode_images(x_ts[i : i + 1])
                encoded_x_ts.append(encoded)
            x_ts = torch.cat(encoded_x_ts, 0).float()
            assert x_ts.shape == (
                self.num_views,
                4,
                self.config.instance_latent_size,
                self.config.instance_latent_size,
            )
        return x_ts

    def inverse_mapping(
        self,
        x_ts,
        xy_map_stack=None,
        mask_stack=None,
        decoding=False,
        voronoi=None,
        num_channels=None,
        canonical_h=None,
        canonical_w=None,
        **kwargs,
    ):
        if voronoi is None:
            voronoi = self.config.voronoi

        x_ts = x_ts.to(self.device)
        if num_channels is None:
            num_channels = 3 if self.config.average_rgb else 4
        if canonical_h is None:
            canonical_h = (
                self.config.canonical_rgb_h
                if self.config.average_rgb
                else self.config.canonical_latent_h
            )
        if canonical_w is None:
            canonical_w = (
                self.config.canonical_rgb_w
                if self.config.average_rgb
                else self.config.canonical_latent_w
            )

        pano = torch.zeros(1, num_channels, canonical_h, canonical_w).to(x_ts)
        pano_count = torch.zeros(1, 1, canonical_h, canonical_w, device=self.device)
        if decoding:
            decoded_x_ts = []
            for i in range(self.num_views):
                decoded = self.decode_latents(x_ts[i : i + 1])
                decoded_x_ts.append(decoded)
            x_ts = torch.cat(decoded_x_ts, 0).float()
            assert x_ts.shape == (
                self.num_views,
                3,
                self.config.instance_rgb_size,
                self.config.instance_rgb_size,
            ), f"decoded_x_ts: {x_ts.shape}"

        for i in range(self.num_views):
            z_i, mask_i = self.inverse_ft(x_ts[i : i + 1], i, pano, xy_map_stack)

            if voronoi:
                # Voronoi filling
                tmp = voronoi_solve(z_i[0].permute(1, 2, 0), mask_i[0, 0])
                z_i = tmp.permute(2, 0, 1).unsqueeze(0)
                mask_i = mask_stack[i].unsqueeze(0).unsqueeze(0)
                z_i = z_i * mask_i
            pano = pano + z_i
            pano_count = pano_count + mask_i

        z_t = pano / (pano_count + 1e-8)

        return z_t

    def compute_noise_preds(
        self, 
        xts, 
        timestep, 
        **kwargs
    ):
        
        return self.model.compute_noise_preds(xts, timestep, **kwargs)
    

    @torch.no_grad()
    def __call__(
        self,
        depth_data_path=None,
        prompt=None,
        negative_prompt=None,
    ):
        """
        Process reverse diffusion steps
        """

        if negative_prompt is None:
            negative_prompt = self.config.negative_prompt
            self.config.negative_prompt = negative_prompt

        prompt = f"Best quality, extremely detailed {self.config.prompt}"
        self.config.prompt = prompt
        

        log_opt = vars(self.config)
        config_path = os.path.join(self.output_dir, "pano_run_config.yaml")
        with open(config_path, "w") as f:
            json.dump(log_opt, f, indent=4)

        num_timesteps = self.model.scheduler.config.num_train_timesteps
        ref_attention_end = self.config.ref_attention_end
        multiview_diffusion_end = self.config.mvd_end
        callback = None

        callback_steps = 1

        if self.config.model == "controlnet":
            if depth_data_path is None:
                depth_data_path = self.config.depth_data_path
            conditioning_images = self.get_depth_conditioning_images(depth_data_path)

            control_guidance_start = self.config.control_guidance_start
            control_guidance_end = self.config.control_guidance_end

            controlnet = (
                self.model.controlnet._orig_mod
                if is_compiled_module(self.model.controlnet)
                else self.model.controlnet
            )
            if not isinstance(control_guidance_start, list) and isinstance(
                control_guidance_end, list
            ):
                control_guidance_start = len(control_guidance_end) * [
                    control_guidance_start
                ]
            elif not isinstance(control_guidance_end, list) and isinstance(
                control_guidance_start, list
            ):
                control_guidance_end = len(control_guidance_start) * [
                    control_guidance_end
                ]
            elif not isinstance(control_guidance_start, list) and not isinstance(
                control_guidance_end, list
            ):
                mult = 1
                control_guidance_start, control_guidance_end = mult * [
                    control_guidance_start
                ], mult * [control_guidance_end]
            controlnet_conditioning_scale = self.config.conditioning_scale

            controlnet_keep = []
            timesteps = self.model.scheduler.timesteps
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(
                    keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
                )

        else:
            controlnet_conditioning_scale = None
            conditioning_images = None
            controlnet_keep = None

        num_images_per_prompt = 1
        if prompt is not None and isinstance(prompt, list):
            assert (
                len(prompt) == 1 and len(negative_prompt) == 1
            ), "Only implemented for 1 (negative) prompt"
        assert num_images_per_prompt == 1, "Only implemented for 1 image per-prompt"
        batch_size = self.num_views

        device = self.device
        guidance_scale = self.config.guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0

        guess_mode = False
        cross_attention_kwargs = None

        prompt_embeds = self.model._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )
        if do_classifier_free_guidance:
            negative_prompt_embeds, prompt_embeds = torch.chunk(prompt_embeds, 2)
        else:
            negative_prompt_embeds = None
                
        # 5. Prepare timesteps
        num_inference_steps = self.config.steps
        self.model.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.model.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.model.unet.config.in_channels
        generator = torch.manual_seed(self.config.seed)

        case_name = METHOD_MAP[str(self.config.case_num)]
        instance_denoising_cases = list(INSTANCE_DENOISING_XT.values()) + list(JOINT_DENOISING_XT.values())
        canonical_denoising_cases = list(CANONICAL_DENOISING_ZT.values()) + list(JOINT_DENOISING_ZT.values())

        get_latent = self.model.prepare_latents
        canonical_latent_h_param = self.config.canonical_latent_h * 8
        canonical_latent_w_param = self.config.canonical_latent_w * 8
        
        func_params = {
            "guidance_scale": guidance_scale,
            "positive_prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "group_metas": self.group_metas,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "guess_mode": guess_mode,
            "ref_attention_end": ref_attention_end,
            "num_timesteps": num_timesteps,
            "cross_attention_kwargs": cross_attention_kwargs,
            "conditioning_images": conditioning_images,
            "controlnet_keep": controlnet_keep,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "generator": generator,
            "predicted_variance": None,
            "cos_weighted": True,
        }
        
        if case_name in instance_denoising_cases:
            if self.config.initialize_xt_from_zt:
                print("[*] Although instance denoising, initialize xT from zT")
                zT = get_latent(
                    1,
                    num_channels_latents,
                    canonical_latent_h_param,
                    canonical_latent_w_param,
                    prompt_embeds.dtype,
                    device,
                    generator,
                )
                
                xts = self.forward_mapping(zT, 
                                           xy_map_stack=self.latent_xy_map_stack,
                                           encoding=False)
                zts = None
                del zT

            else:
                xts = get_latent(
                    batch_size,
                    num_channels_latents,
                    self.config.instance_latent_size * 8,
                    self.config.instance_latent_size * 8,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    None,
                )  # (B, 4, 96, 96)
                zts = None
        else:
            zts = get_latent(
                1,
                num_channels_latents,
                self.config.canonical_latent_h * 8,
                self.config.canonical_latent_w * 8,
                prompt_embeds.dtype,
                device,
                generator,
            )
            xts = None

        input_params = {"zts": zts, "xts": xts}

        mapping_dict = {
            "encoding": self.config.average_rgb,
            "decoding": self.config.average_rgb,
        }
        if self.config.average_rgb:
            mapping_dict["xy_map_stack"] = self.rgb_xy_map_stack
            mapping_dict["mask_stack"] = self.rgb_mask_stack
        else:
            mapping_dict["xy_map_stack"] = self.latent_xy_map_stack
            mapping_dict["mask_stack"] = self.latent_mask_stack

        eta = 0.0
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.model.scheduler.order
        )
        
        func_params.update(mapping_dict)

        alphas = self.model.scheduler.alphas_cumprod ** (0.5)
        sigmas = (1 - self.model.scheduler.alphas_cumprod) ** (0.5)
        with self.model.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                func_params["cur_index"] = i
                self.model.set_up_coefficients(t, self.config.sampling_method)

                if t > (1 - multiview_diffusion_end) * num_timesteps:
                    out_params = self.one_step_process(
                        input_params=input_params,
                        timestep=t,
                        alphas=alphas,
                        sigmas=sigmas,
                        case_name=case_name,
                        **func_params,
                    )

                    if case_name in instance_denoising_cases:
                        input_params["xts"] = out_params["x_t_1"]
                        input_params["zts"] = None
                        log_x_prevs = out_params["x_t_1"]
                        log_x0s = out_params["x0s"]
                    elif case_name in canonical_denoising_cases:
                        input_params["xts"] = None
                        input_params["zts"] = out_params["z_t_1"]
                        log_x_prevs = self.forward_mapping(
                            out_params["z_t_1"], **mapping_dict
                        )
                        log_x0s = self.forward_mapping(
                            out_params["z0s"], **mapping_dict
                        )
                else:
                    if case_name in instance_denoising_cases:
                        latents = out_params["x_t_1"]

                    elif case_name in canonical_denoising_cases:
                        if out_params.get("x_t_1") is None:
                            assert (
                                out_params.get("z_t_1") is not None
                            ), f"{out_params['z_t_1']}"
                            latents = self.forward_mapping(
                                out_params["z_t_1"], **mapping_dict
                            )
                        else:
                            latents = out_params["x_t_1"]

                    noise_pred = self.compute_noise_preds(
                        latents, t, **func_params
                    )
                    step_results = self.model.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=True
                    )

                    pred_original_sample = step_results["pred_original_sample"]
                    latents = step_results["prev_sample"]
                    out_params = dict()
                    out_params["x_t_1"] = latents
                    out_params["x0s"] = pred_original_sample

                    log_x_prevs = out_params["x_t_1"]
                    log_x0s = out_params["x0s"]

                if (i + 1) % self.config.log_interval == 0:
                    self.intermediate_dir = self.output_dir / f"intermediate/{i}"
                    self.intermediate_dir.mkdir(exist_ok=True, parents=True)

                    log_x_prev_img = self.instance_latents_to_pano_image(log_x_prevs)
                    log_x0_img = self.instance_latents_to_pano_image(log_x0s)

                    log_img = merge_images([log_x_prev_img, log_x0_img])
                    log_img.save(f"{self.intermediate_dir}/i={i}_t={t}.png")

                    for view_idx, log_x0 in enumerate(log_x0s[:10]):
                        decoded = self.decode_latents(log_x0.unsqueeze(0)).float()
                        TF.to_pil_image(decoded[0].cpu()).save(
                            f"{self.intermediate_dir}/i={i}_v={view_idx}_view.png"
                        )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.model.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

            if case_name in instance_denoising_cases:
                final_out = self.forward_mapping(
                    self.inverse_mapping(out_params["x_t_1"], **mapping_dict),
                    **mapping_dict,
                )
            elif case_name in canonical_denoising_cases:
                if multiview_diffusion_end == 1:
                    final_out = self.forward_mapping(
                        out_params["z_t_1"], **mapping_dict
                    )
                else:
                    final_out = self.forward_mapping(
                        self.inverse_mapping(out_params["x_t_1"], **mapping_dict),
                        **mapping_dict,
                    )

            self.result_dir = f"{self.output_dir}/results"
            os.makedirs(self.result_dir, exist_ok=True)
            
            final_img = self.instance_latents_to_pano_image(final_out)
            final_img.save(f"{self.result_dir}/final.png")

            equ = Equirectangular(f"{self.result_dir}/final.png")
            for i, theta in enumerate(np.random.randint(0, 360, 10)):
                pers_img = equ.GetPerspective(60, theta, 0, 512, 512)[..., [2, 1, 0]]
                pers_img = Image.fromarray(pers_img)
                pers_img.save(
                    f"{self.result_dir}/final_pers_sample_{i}_theta={theta}.png"
                )

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = latents.half()
        latents = 1 / 0.18215 * latents
        imgs = self.model.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def encode_images(self, imgs):
        imgs = imgs.half()
        imgs = (imgs - 0.5) * 2
        posterior = self.model.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    @torch.no_grad()
    def instance_latents_to_pano_image(self, xs):
        """
        Input:
            x: [N,C,h,w]
        Otuput:
            panorama img
        """
        pano = self.inverse_mapping(
            xs,
            self.rgb_xy_map_stack,
            self.rgb_mask_stack,
            num_channels=3,
            canonical_h=self.config.canonical_rgb_h,
            canonical_w=self.config.canonical_rgb_w,
            decoding=True,
        )
        return TF.to_pil_image(pano[0].cpu())

    def get_depth_conditioning_images(self, depth_data_path):
        def recon_pil_img(pil_img, recon_save_path):
            tensor = TF.to_tensor(pil_img).to(self.device)[None]
            instances = self.forward_mapping(
                tensor, self.rgb_xy_map_stack, encoding=False
            )
            recon = self.inverse_mapping(
                instances,
                self.rgb_xy_map_stack,
                self.rgb_mask_stack,
                decoding=False,
                num_channels=3,
                canonical_h=self.config.canonical_rgb_h,
                canonical_w=self.config.canonical_rgb_w,
            )
            recon_img = TF.to_pil_image(recon[0].cpu())
            recon_img.save(recon_save_path)
            return instances

        disparity_img = Image.open(
            str(depth_data_path).replace("_depth.dpt", "_vis.png")
        ).convert("RGB")
        disp_instances = recon_pil_img(
            disparity_img, f"{self.output_dir}/disparity_recon.png"
        )

        disp_np = np.asarray(disparity_img)
        depth_img = (
            Image.fromarray((disp_np * 255).astype(np.uint8))
            .convert("L")
            .convert("RGB")
        )
        depth_instances = recon_pil_img(
            depth_img, f"{self.output_dir}/cond_depth_recon.png"
        )

        rgb_img = Image.open(
            str(depth_data_path).replace("_depth.dpt", "_rgb.png")
        ).convert("RGB")
        rgb_instances = recon_pil_img(rgb_img, f"{self.output_dir}/rgb_recon.png")

        del disp_instances
        del rgb_instances

        return depth_instances.half()
