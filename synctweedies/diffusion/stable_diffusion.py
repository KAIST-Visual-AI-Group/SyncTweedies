import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from synctweedies.diffusion.diffusion_base import DiffusionModel
from synctweedies.utils.mesh_utils import replace_attention_processors, SamplewiseAttnProcessor2_0


class SyncTweediesSD(DiffusionModel, StableDiffusionPipeline):
    def __init__(
        self, 
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        
        super().__init__(
            vae, text_encoder, tokenizer, unet, 
            scheduler, safety_checker, 
            feature_extractor, requires_safety_checker
        )
        
        self.scheduler = scheduler
        self.enable_model_cpu_offload()
        self.enable_vae_slicing()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        

    def compute_noise_preds(
        self, 
        xts, 
        ts, 
        prompt_embeds,
        group_metas,
        num_timesteps,
        ref_attention_end,
        guidance_scale,
        **kwargs,
    ):
        
        C, H, W = xts.shape[-3], xts.shape[-2], xts.shape[-1]
        xts = xts.reshape(-1, C, H, W)

        model_input_batches = [torch.index_select(xts, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in group_metas]
        noise_pred_dict = {}
        for mode, prompt in zip(["uncond", "text"], prompt_embeds):
            noise_pred_list = []
            for model_input_batch, meta in zip (model_input_batches, group_metas):
                xt_input = model_input_batch
                xt_input = self.scheduler.scale_model_input(xt_input, ts)
                prompt_embeds_batch = [prompt] * len(meta[0])
                prompt_embeds_batch = torch.stack(prompt_embeds_batch, dim=0)

                if ts > num_timesteps * (1- ref_attention_end):
                    replace_attention_processors(self.unet, SamplewiseAttnProcessor2_0, attention_mask=meta[2], ref_attention_mask=meta[3], ref_weight=1)
                else:
                    replace_attention_processors(self.unet, SamplewiseAttnProcessor2_0, attention_mask=meta[2], ref_attention_mask=meta[3], ref_weight=0)
                
                noise_preds = self.unet(
                    xt_input.half(),
                    ts,
                    encoder_hidden_states=prompt_embeds_batch,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]
                noise_pred_list.append(noise_preds)
            noise_pred_dict[mode] = torch.cat(noise_pred_list, dim=0)

        noise_preds_stack = noise_pred_dict["uncond"] + guidance_scale * (noise_pred_dict["text"] - noise_pred_dict["uncond"])

        return noise_preds_stack