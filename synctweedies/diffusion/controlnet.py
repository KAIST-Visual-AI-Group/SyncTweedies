import torch
from typing import List, Tuple, Union


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from synctweedies.diffusion.diffusion_base import DiffusionModel
from synctweedies.utils.mesh_utils import replace_attention_processors, SamplewiseAttnProcessor2_0


class SyncTweediesControlNet(DiffusionModel, StableDiffusionControlNetPipeline):
    def __init__(
        self, 
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = False,
        gpu_id=0,
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, 
            controlnet, scheduler, safety_checker, 
            feature_extractor, requires_safety_checker
        )

        self.scheduler = scheduler
        self.model_cpu_offload_seq = "vae->text_encoder->unet->vae"
        self.enable_model_cpu_offload()
        self.enable_vae_slicing()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)


    def compute_noise_preds(
        self, 
        x_t, 
        t,
        guidance_scale,
        positive_prompt_embeds,
        negative_prompt_embeds,
        group_metas,
        do_classifier_free_guidance,
        ref_attention_end,
        num_timesteps,
        cross_attention_kwargs,
        cur_index,
        conditioning_images,
        controlnet_keep,
        controlnet_conditioning_scale,
        **kwargs,
    ):
        """
        x_t -> epsilon_t
        """
        device = x_t.device
        latent_model_input = self.scheduler.scale_model_input(x_t, t).type(
            self.unet.dtype
        )

        prompt_embeds_groups = {"positive": positive_prompt_embeds}
        result_groups = {}
        if do_classifier_free_guidance:
            prompt_embeds_groups["negative"] = negative_prompt_embeds

        for prompt_tag, prompt_embeds in prompt_embeds_groups.items():
            control_model_input = latent_model_input
            if isinstance(controlnet_keep[cur_index], list):
                cond_scale = [
                    c * s
                    for c, s in zip(
                        controlnet_conditioning_scale, controlnet_keep[cur_index]
                    )
                ]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[cur_index]

            down_block_res_samples_list = []
            mid_block_res_sample_list = []

            model_input_batches = [
                torch.index_select(
                    control_model_input,
                    dim=0,
                    index=torch.tensor(meta[0], device=device),
                )
                for meta in group_metas
            ]
            conditioning_images_batches = [
                torch.index_select(
                    conditioning_images,
                    dim=0,
                    index=torch.tensor(meta[0], device=device),
                )
                for meta in group_metas
            ]
            
            if prompt_embeds.shape[0] != control_model_input.shape[0]:
                prompt_embeds_batch = [prompt_embeds] * len(group_metas[0][0])
                prompt_embeds_batch = torch.cat(prompt_embeds_batch, 0)

            else:
                prompt_embeds_batch = prompt_embeds

            for model_input_batch, conditioning_images_batch, meta in zip(
                model_input_batches, conditioning_images_batches, group_metas
            ):
                (
                    down_block_res_samples,
                    mid_block_res_sample,
                ) = self.controlnet(
                    model_input_batch,
                    t,
                    encoder_hidden_states=prompt_embeds_batch,
                    controlnet_cond=conditioning_images_batch,
                    conditioning_scale=cond_scale,
                    return_dict=False,
                )
                down_block_res_samples_list.append(down_block_res_samples)
                mid_block_res_sample_list.append(mid_block_res_sample)

            noise_pred_list = []
            model_input_batches = [
                torch.index_select(
                    latent_model_input,
                    dim=0,
                    index=torch.tensor(meta[0], device=device),
                )
                for meta in group_metas
            ]

            for (
                model_input_batch,
                down_block_res_samples_batch,
                mid_block_res_sample_batch,
                meta,
            ) in zip(
                model_input_batches,
                down_block_res_samples_list,
                mid_block_res_sample_list,
                group_metas,
            ):
                if t > num_timesteps * (1 - ref_attention_end):
                    replace_attention_processors(
                        self.unet,
                        SamplewiseAttnProcessor2_0,
                        attention_mask=meta[2],
                        ref_attention_mask=meta[3],
                        ref_weight=1,
                    )
                else:
                    replace_attention_processors(
                        self.unet,
                        SamplewiseAttnProcessor2_0,
                        attention_mask=meta[2],
                        ref_attention_mask=meta[3],
                        ref_weight=0,
                    )

                noise_pred = self.unet(
                    model_input_batch,
                    t,
                    encoder_hidden_states=prompt_embeds_batch,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples_batch,
                    mid_block_additional_residual=mid_block_res_sample_batch,
                    return_dict=False,
                )[0]
                noise_pred_list.append(noise_pred)

            noise_pred_list = [
                torch.index_select(
                    noise_pred,
                    dim=0,
                    index=torch.tensor(meta[1], device=device),
                )
                for noise_pred, meta in zip(noise_pred_list, group_metas)
            ]
            noise_pred = torch.cat(noise_pred_list, dim=0)
            noise_pred_list = None
            model_input_batches = prompt_embeds_batches = None

            result_groups[prompt_tag] = noise_pred

        positive_noise_pred = result_groups["positive"]

        if do_classifier_free_guidance:
            noise_pred = result_groups["negative"] + guidance_scale * (
                positive_noise_pred - result_groups["negative"]
            )

        del result_groups

        return noise_pred
    
    def set_up_coefficients(self, t, sampling_method):
        if sampling_method == "ddim":
            self.prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        elif sampling_method == "ddpm":
            self.prev_t = self.scheduler.previous_timestep(t)
            
        self.alpha_prod_t = self.scheduler.alphas_cumprod[t]
        self.alpha_prod_t_prev = self.scheduler.alphas_cumprod[self.prev_t] if self.prev_t >= 0 else torch.tensor(1.0)
        self.beta_prod_t = 1 - self.alpha_prod_t
        self.beta_prod_t_prev = 1 - self.alpha_prod_t_prev
        self.current_alpha_t = self.alpha_prod_t / self.alpha_prod_t_prev
        self.current_beta_t = 1 - self.current_alpha_t
    

    