from abc import *
import torch
from datetime import datetime
from pathlib import Path
import os

from diffusers import (ControlNetModel, DDIMScheduler, DDPMScheduler,
                       StableDiffusionControlNetPipeline,
                       DiffusionPipeline, StableDiffusionPipeline)
from diffusers.utils import randn_tensor

from synctweedies.diffusion.controlnet import SyncTweediesControlNet
from synctweedies.diffusion.stable_diffusion import SyncTweediesSD

def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


class BaseModel(metaclass=ABCMeta):
    def __init__(self):
        self.init_model()
        self.init_mapper()
        
    def initialize(self):
        now = get_current_time()
        save_top_dir = self.config.save_top_dir
        tag = self.config.tag
        save_dir_now = self.config.save_dir_now 
        
        if save_dir_now:
            self.output_dir = Path(save_top_dir) / f"{tag}/{now}"
        else:
            self.output_dir = Path(save_top_dir) / f"{tag}"
        
        if not os.path.isdir(self.output_dir):
            self.output_dir.mkdir(exist_ok=True, parents=True)
        else:
            print(f"Results exist in the output directory, use time string to avoid name collision.")
            exit(0)
            
        print("[*] Saving at ", self.output_dir)

    @abstractmethod
    def init_mapper(self, **kwargs):
        self.mapper = None

    def init_model(self, **kwargs):
        """
        Load diffusion model
        """

        def get_scheduler(pipe):
            if self.config.sampling_method == "ddpm":
                _scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            elif self.config.sampling_method == "ddim":
                _scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            else:
                raise NotImplementedError(f"{self.config.sampling_method} not implemented")
            return _scheduler

        if self.config.model == "controlnet":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                variant="fp16",
                torch_dtype=torch.float16,
            ).to(self.device)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
            )
            pipe.scheduler = get_scheduler(pipe)
            self.model = SyncTweediesControlNet(
                gpu_id=self.config.gpu, **pipe.components
            )

        elif self.config.model == "deepfloyd":
            self.stage_1 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0", 
                variant="fp16", 
                torch_dtype=torch.float16,
            )
            self.stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            )
            
            scheduler = get_scheduler(self.stage_1)
            self.stage_1.scheduler = self.stage_2.scheduler = scheduler
            

        elif self.config.model == "sd":
            pipe = StableDiffusionPipeline.from_pretrained(
                self.config.sd_path,
                torch_dtype=torch.float16,
            )
            pipe.scheduler = get_scheduler(pipe)
            self.model = SyncTweediesSD(
                **pipe.components,
            )
            
        else:
            raise NotImplementedError(f"{self.config.model}")

        if self.config.model in ["sd", "controlnet"]:
            self.model.text_encoder.requires_grad_(False)
            self.model.unet.requires_grad_(False)
            if hasattr(self.model, "vae"):
                self.model.vae.requires_grad_(False)
        else:
            self.stage_1.text_encoder.requires_grad_(False)
            self.stage_2.unet.requires_grad_(False)
            self.stage_2.unet.requires_grad_(False)
            
            self.stage_1 = self.stage_1.to(self.device)
            self.stage_2 = self.stage_2.to(self.device)

    @abstractmethod
    def forward_mapping(self, z_t: torch.FloatTensor, **kwargs):
        """
        Return {x_t^i} from z_t
        Input:
            z_t: [B,*]
        Output:
            {x_t^i}_{i=1}^N: [B,N,*], where N denotes # of variables.
        """
        pass

    @abstractmethod
    def inverse_mapping(self, x_ts, **kwargs):
        """
        Return z_t from {x_t^i}_{i=1}^N
        Input:
            x_ts: {x_t^i}. [B,N,*]
        Output:
            z_t: [B,*]
        """
        pass

    def compute_tweedie(self, xts, eps, timestep, alphas, sigmas, **kwargs):
        """
        Input:
            xts, eps: [B,*]
            timestep: [B]
            x_t = alpha_t * x0 + sigma_t * eps
        Output:
            pred_x0s: [B,*]
        """
        if eps.shape[-3] == xts.shape[-3] * 2:
            eps, predicted_variance = torch.split(eps, xts.shape[-3], dim=-3)
        else:
            predicted_variance = None

        assert xts.shape == eps.shape

        alpha_t = alphas[timestep]
        sigma_t = sigmas[timestep]

        pred_x0s = (xts - sigma_t * eps) / alpha_t
        assert pred_x0s.shape == xts.shape
        return pred_x0s

    
    def compute_prev_state(
        self, xts, pred_x0s, timestep, eps, eta=0, **kwargs
    ):
        """
        Input:
            xts: [N,C,H,W]
        Output:
            pred_prev_sample: [N,C,H,W]
        """
        device = xts.device
        generator = kwargs.get("generator")

        orig_xts_shape = xts.shape
        C, H, W = xts.shape[-3], xts.shape[-2], xts.shape[-1]
        xts = xts.reshape(-1, C, H, W)
        pred_x0s = pred_x0s.reshape(-1, C, H, W)
        
        if self.config.app == "ambiguous_image":
            scheduler = self.stage_1.scheduler
        else:
            scheduler = self.model.scheduler

        if eps is not None and eps.shape[-3] == xts.shape[-3] * 2:
            eps = eps.reshape(-1, 2 * C, H, W)
            if scheduler.variance_type in [
                "learned",
                "learned_range",
            ]:
                eps, predicted_variance = torch.split(eps, xts.shape[-3], dim=-3)
            else:
                predicted_variance = None
        else:
            predicted_variance = None

        prev_timestep = (
            timestep
            - scheduler.config.num_train_timesteps
            // scheduler.num_inference_steps
        )

        t = timestep
        prev_t = prev_timestep

        # 1. compute alphas, betas
        if self.config.sampling_method == "ddpm":
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[prev_t]
                if prev_t >= 0
                else scheduler.one
            )
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t

            # 3. Clip or threshold "predicted x_0"
            if scheduler.config.thresholding:
                pred_x0s = scheduler._threshold_sample(pred_x0s)
            elif scheduler.config.clip_sample:
                pred_x0s = pred_x0s.clamp(
                    -scheduler.config.clip_sample_range,
                    scheduler.config.clip_sample_range,
                )

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (
                alpha_prod_t_prev ** (0.5) * current_beta_t
            ) / beta_prod_t
            current_sample_coeff = (
                current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
            )

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_prev_sample = (
                pred_original_sample_coeff * pred_x0s + current_sample_coeff * xts
            )

            # 6. Add noise
            variance = 0
            if t > 0:
                variance_noise = randn_tensor(
                    pred_prev_sample.shape,
                    generator=generator,
                    device=device,
                    dtype=xts.dtype,
                )
                if predicted_variance is None:
                    variance = (
                        scheduler._get_variance(
                            t,
                            predicted_variance=predicted_variance,
                            variance_type="fixed_small_log",
                        )
                        * variance_noise
                    )
                elif scheduler.variance_type == "fixed_small_log":
                    variance = (
                        scheduler._get_variance(
                            t, predicted_variance=predicted_variance
                        )
                        * variance_noise
                    )
                elif scheduler.variance_type == "learned_range":
                    variance = scheduler._get_variance(
                        t, predicted_variance=predicted_variance
                    )
                    variance = torch.exp(0.5 * variance) * variance_noise
                else:
                    variance = (
                        scheduler._get_variance(
                            t, predicted_variance=predicted_variance
                        )
                        ** 0.5
                    ) * variance_noise

            pred_prev_sample = pred_prev_sample + variance
            pred_prev_sample = pred_prev_sample.reshape(orig_xts_shape)
            
        elif self.config.sampling_method == "ddim":
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[prev_t]
                if prev_t >= 0
                else scheduler.final_alpha_cumprod
            )

            def _get_variance(scheduler, timestep, prev_timestep):
                alpha_prod_t = scheduler.alphas_cumprod[timestep]
                alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev

                variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

                return variance
            
            variance = _get_variance(scheduler, t, prev_t)
            std_dev_t = eta * variance ** (0.5)

            xt_coeff = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) / torch.sqrt(1 - alpha_prod_t)
            x0_coeff = torch.sqrt(alpha_prod_t_prev) - torch.sqrt(alpha_prod_t) / torch.sqrt(1 - alpha_prod_t) * torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2)

            pred_prev_sample = xt_coeff * xts + x0_coeff * pred_x0s
            if eta > 0:
                variance_noise = randn_tensor(
                    pred_prev_sample.shape,
                    generator=generator,device=device,dtype=xts.dtype)
                variance = std_dev_t * variance_noise
                pred_prev_sample = pred_prev_sample + variance_noise

        return pred_prev_sample

    @abstractmethod
    def compute_noise_preds(self, xts, ts, **kwargs):
        """
        Input:
            xts: [B,*]
            ts: [B]
        Output:
            noise_preds: [B,*]
        """
        raise NotImplementedError("Not implemented")


    def one_step_process(
        self, input_params, timestep, alphas, sigmas, case_name: str, **kwargs
    ):
        """
        Input:
            latents: either xt or zt. [B,*]
        Output:
            output: the same with latent.
        """

        if case_name == "0":  # updated
            xts = input_params["xts"]

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "1":  # updated
            xts = input_params["xts"]

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "2": # updated
            xts = input_params["xts"]

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "3": # updated
            xts = input_params["xts"]

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)
            z_t_1 = self.inverse_mapping(x_t_1, var_type="latent", **kwargs)
            x_t_1 = self.forward_mapping(z_t_1, bg=x_t_1, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "4": # updated
            zts = input_params["zts"]
            xts = self.forward_mapping(zts, bg=input_params["xts"] ,**kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)

            z0s = self.compute_tweedie(
                zts, delta_preds, timestep, alphas, sigmas, **kwargs
            )
            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)

            out_params = {
                "x0s": None,
                "z0s": z0s,
                "x_t_1": None,
                "z_t_1": z_t_1,
            }

            return out_params

        elif case_name == "5": # updated
            zts = input_params["zts"]
            xts = self.forward_mapping(zts, bg=input_params["xts"], **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(
                eps_preds, var_type="eps", **kwargs
            )  # for z_t_1 computation.

            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)

            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)

            out_params = {
                "x0s": None,
                "z0s": z0s,
                "x_t_1": None,
                "z_t_1": z_t_1,
            }

            return out_params

        elif case_name == "6": # updated
            zts = input_params["zts"]
            xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)

            x0s = self.compute_tweedie(xts, eps_preds, timestep, alphas, sigmas, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)
            z_t_1 = self.inverse_mapping(x_t_1, var_type="latent", **kwargs)

            # for completeness
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)

            out_params = {
                "x0s": None,
                "z0s": z0s,
                "x_t_1": None,
                "z_t_1": z_t_1,
            } 
            return out_params

        elif case_name == "7": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "8": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "9": # updated
            xts = input_params["xts"]

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)

            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "10": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "11": # updated
            xts = input_params["xts"]

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "12": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "13": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=xts, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "14": # updated
            xts = input_params["xts"]

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "15": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "16": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "17": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "18": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "19": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "20": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "21": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "22": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "23": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "24": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "25": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "26": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "27": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "28": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "29": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "30": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "31": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "32": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "33": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "34": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, bg=eps_preds, **kwargs)

            x0s = self.compute_tweedie(
                hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, bg=x0s, **kwargs)

            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "35": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)

            z0s = self.compute_tweedie(
                zts, delta_preds, timestep, alphas, sigmas, **kwargs
            )
            x0s = self.forward_mapping(z0s, bg=xts, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "36": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)

            z0s = self.compute_tweedie(
                zts, delta_preds, timestep, alphas, sigmas, **kwargs
            )
            x0s = self.forward_mapping(z0s, bg=xts, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "37": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)

            z0s = self.compute_tweedie(
                zts, delta_preds, timestep, alphas, sigmas, **kwargs
            )
            x0s = self.forward_mapping(z0s, bg=xts, **kwargs)

            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "38": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, bg=xts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)

            z0s = self.compute_tweedie(
                zts, delta_preds, timestep, alphas, sigmas, **kwargs
            )
            x0s = self.forward_mapping(z0s, bg=xts, **kwargs)

            x_t_1 = self.compute_prev_state(hat_xts, x0s, timestep, eps_preds, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "39": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            
            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)

            z0s = self.compute_tweedie(zts, delta_preds, timestep, alphas, sigmas, **kwargs)
            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)
            x_t_1 = self.forward_mapping(z_t_1, **kwargs)

            # for completeness. actually, the terms below are not used in computation.
            x0s = self.compute_tweedie(xts, eps_preds, timestep, alphas, sigmas, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        ###########
        elif case_name == "40": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, **kwargs)
            
            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)

            z0s = self.compute_tweedie(zts, delta_preds, timestep, alphas, sigmas, **kwargs)
            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)
            x_t_1 = self.forward_mapping(z_t_1, **kwargs)

            # for completeness. actually, the terms below are not used in computation.
            x0s = self.compute_tweedie(xts, eps_preds, timestep, alphas, sigmas, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }

            return out_params

        elif case_name == "41": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            x0s = self.compute_tweedie(xts, eps_preds, timestep, alphas, sigmas, **kwargs)
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)

            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs) # for z_t_1 computation.
            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)

            x_t_1 = self.forward_mapping(z_t_1, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }
            return out_params
    

        elif case_name == "42": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            x0s = self.compute_tweedie(xts, eps_preds, timestep, alphas, sigmas, **kwargs)
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)

            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs) # for z_t_1 computation.
            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)

            x_t_1 = self.forward_mapping(z_t_1, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }
            return out_params
        elif case_name == "43": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs) # for z_t_1 computation.
            eps_preds = self.forward_mapping(delta_preds, **kwargs)

            x0s = self.compute_tweedie(xts, eps_preds, timestep, alphas, sigmas, **kwargs)
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)

            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)

            x_t_1 = self.forward_mapping(z_t_1, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }
            return out_params

        elif case_name == "44": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs) # for z_t_1 computation.
            eps_preds = self.forward_mapping(delta_preds, **kwargs)

            x0s = self.compute_tweedie(xts, eps_preds, timestep, alphas, sigmas, **kwargs)
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)

            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)

            x_t_1 = self.forward_mapping(z_t_1, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }
            return out_params
        elif case_name == "45": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)

            x0s = self.compute_tweedie(hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs)
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)

            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs) # for z_t_1 computation.
            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)

            x_t_1 = self.forward_mapping(z_t_1, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }
            return out_params

        elif case_name == "46": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)

            x0s = self.compute_tweedie(hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs)
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)

            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs) # for z_t_1 computation.
            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)
            x_t_1 = self.forward_mapping(z_t_1, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }
            return out_params
        
        elif case_name == "47": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, **kwargs)

            x0s = self.compute_tweedie(hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs)
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)

            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)
            x_t_1 = self.forward_mapping(z_t_1, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }
            return out_params
 
        elif case_name == "48": # updated
            xts = input_params["xts"]
            zts = self.inverse_mapping(xts, var_type="latent", **kwargs)
            hat_xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(hat_xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, **kwargs)

            x0s = self.compute_tweedie(hat_xts, eps_preds, timestep, alphas, sigmas, **kwargs)
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            
            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)
            x_t_1 = self.forward_mapping(z_t_1, **kwargs)

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": None,
            }
            return out_params

        elif case_name == "49": #updated
            zts = input_params["zts"]
            xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, **kwargs)

            x0s = self.compute_tweedie(
                xts, eps_preds, timestep, alphas, sigmas, **kwargs
            )
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)

            z_t_1 = self.compute_prev_state(zts, z0s, timestep, delta_preds, **kwargs)

            out_params = {
                "x0s": None,
                "z0s": z0s,
                "x_t_1": None,
                "z_t_1": z_t_1,
            }
            return out_params
       
        ############
        elif case_name == "50": # updated
            zts = input_params["zts"]
            xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, **kwargs)

            x0s = self.compute_tweedie(xts, eps_preds, timestep, alphas, sigmas, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)
            z_t_1 = self.inverse_mapping(x_t_1, var_type="latent", **kwargs)

            # for completeness
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)

            out_params = {
                "x0s": None,
                "z0s": z0s,
                "x_t_1": None,
                "z_t_1": z_t_1,
            } 
            return out_params

        elif case_name == "51": # updated
            zts = input_params["zts"]
            xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)

            x0s = self.compute_tweedie(xts, eps_preds, timestep, alphas, sigmas, **kwargs)
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)
            z_t_1 = self.inverse_mapping(x_t_1, var_type="latent", **kwargs)

            out_params = {
                "x0s": None,
                "z0s": z0s,
                "x_t_1": None,
                "z_t_1": z_t_1,
            } 
            return out_params
        
        elif case_name == "52": # updated
            zts = input_params["zts"]
            xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)
            eps_preds = self.forward_mapping(delta_preds, **kwargs)

            x0s = self.compute_tweedie(xts, eps_preds, timestep, alphas, sigmas, **kwargs)
            z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs)
            x0s = self.forward_mapping(z0s, **kwargs)
       
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)
            z_t_1 = self.inverse_mapping(x_t_1, var_type="latent", **kwargs)

            out_params = {
                "x0s": None,
                "z0s": z0s,
                "x_t_1": None,
                "z_t_1": z_t_1,
            }
            return out_params

        elif case_name == "53": # updated
            zts = input_params["zts"]
            xts = self.forward_mapping(zts, **kwargs)

            eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            delta_preds = self.inverse_mapping(eps_preds, var_type="eps", **kwargs)

            z0s = self.compute_tweedie(zts, delta_preds, timestep, alphas, sigmas, **kwargs)
            x0s = self.forward_mapping(z0s, **kwargs)

            x_t_1 = self.compute_prev_state(xts, x0s, timestep, eps_preds, **kwargs)
            z_t_1 = self.inverse_mapping(x_t_1, var_type="latent", **kwargs)

            out_params = {
                "x0s": None,
                "z0s": z0s,
                "x_t_1": None,
                "z_t_1": z_t_1,
            }
            return out_params
        
        # Below are combined denoising processes.
        elif case_name == "54":
            # instance
            xts = input_params["xts"]
            instance_eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            instance_delta_preds = self.inverse_mapping(instance_eps_preds, var_type="eps", **kwargs)
            instance_eps_preds = self.forward_mapping(instance_delta_preds, bg=instance_eps_preds, **kwargs)
            
            # canonical 
            zts = input_params["zts"]
            xts_from_zts = self.forward_mapping(zts, **kwargs)
            canonical_eps_preds = self.compute_noise_preds(xts_from_zts, timestep, **kwargs)
            canonical_delta_preds = self.inverse_mapping(canonical_eps_preds, var_type="eps", **kwargs)
            canonical_eps_preds = self.forward_mapping(canonical_delta_preds, bg=canonical_eps_preds, **kwargs)
            
            # use combined term in instance
            combined_delta_preds = (instance_delta_preds + canonical_delta_preds) / 2
            combined_eps_preds = self.forward_mapping(combined_delta_preds, **kwargs)
            
            x0s = self.compute_tweedie(
                xts, combined_eps_preds, timestep, alphas, sigmas, **kwargs
            )
            x_t_1 = self.compute_prev_state(xts, x0s, timestep, combined_eps_preds, **kwargs)
            
            # use canonical term only for canonical denoising
            z0s = self.compute_tweedie(
                zts, canonical_delta_preds, timestep, alphas, sigmas,**kwargs
            )
            z_t_1 = self.compute_prev_state(zts, z0s, timestep, canonical_delta_preds, **kwargs
                )

            out_params = {
                "x0s": x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": z_t_1,
            }
            return out_params
            
        elif case_name == "55":
            # instance
            xts = input_params["xts"]
            instance_eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)

            instance_x0s = self.compute_tweedie(
                xts, instance_eps_preds, timestep, alphas, sigmas, **kwargs
            )
            instance_z0s = self.inverse_mapping(
                instance_x0s, var_type="tweedie", **kwargs
            )
            
            # canonical
            zts = input_params["zts"]
            xts_from_zts = self.forward_mapping(zts, **kwargs)
            canonical_eps_preds = self.compute_noise_preds(xts_from_zts, timestep, **kwargs)
            canonical_delta_preds = self.inverse_mapping(canonical_eps_preds, var_type="eps", **kwargs)

            canonical_z0s = self.compute_tweedie(
                zts, canonical_delta_preds, timestep, alphas, sigmas, **kwargs
            )

            # use combined term in instance
            combined_z0s = (instance_z0s + canonical_z0s) / 2
            combined_x0s = self.forward_mapping(combined_z0s, **kwargs)
            
            x_t_1 = self.compute_prev_state(xts, combined_x0s, timestep, instance_eps_preds, **kwargs) # TODO: The use of instance_eps_preds would not be correct in DDPM sampling.
            
            # use canonical term only for canonical denoising.
            z_t_1 = self.compute_prev_state(zts, canonical_z0s, timestep, canonical_delta_preds, **kwargs
            )
            out_params = {
                "x0s": instance_x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": z_t_1,
            }
            return out_params

        elif case_name == "56":
            # instance
            xts = input_params["xts"]
            instance_eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            instance_x0s = self.compute_tweedie(
                xts, instance_eps_preds, timestep, alphas, sigmas, **kwargs
            )
            instance_x_t_1 = self.compute_prev_state(xts, instance_x0s, timestep, instance_eps_preds, **kwargs)
            instance_z_t_1 = self.inverse_mapping(instance_x_t_1, var_type="latent", **kwargs)

            # canonical
            zts = input_params["zts"]
            xts_from_zts = self.forward_mapping(zts, **kwargs)
            canonical_eps_preds = self.compute_noise_preds(xts_from_zts, timestep, **kwargs)
            canonical_delta_preds = self.inverse_mapping(canonical_eps_preds, var_type="eps", **kwargs)
            canonical_z0s = self.compute_tweedie(
                zts, canonical_delta_preds, timestep, alphas, sigmas, **kwargs
            )
            canonical_z_t_1 = self.compute_prev_state(zts, canonical_z0s, timestep, canonical_delta_preds, **kwargs)
            z_t_1 = canonical_z_t_1

            # use combined term in instance
            combined_z_t_1 = (instance_z_t_1 + canonical_z_t_1) / 2
            combined_x_t_1 = self.forward_mapping(combined_z_t_1, **kwargs)
            x_t_1 = combined_x_t_1

            out_params = {
                "x0s": instance_x0s,
                "z0s": None,
                "x_t_1": x_t_1,
                "z_t_1": z_t_1,
            }
            return out_params
            
        elif case_name == "57":
            # canonical
            zts = input_params["zts"]
            xts_from_zts = self.forward_mapping(zts, **kwargs)
            canonical_eps_preds = self.compute_noise_preds(xts_from_zts, timestep, **kwargs)
            canonical_delta_preds = self.inverse_mapping(canonical_eps_preds, var_type="eps", **kwargs)

            # instance
            xts = input_params["xts"]
            instance_eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            instance_delta_preds = self.inverse_mapping(instance_eps_preds, var_type="eps", **kwargs)

            x0s = self.compute_tweedie(xts, instance_eps_preds, timestep, alphas, sigmas, **kwargs)
            x_t_1 = self.compute_prev_state(
                    xts, x0s, timestep, instance_eps_preds, **kwargs
            )

            # use combined term in canonical
            combined_delta_preds = (canonical_delta_preds + instance_delta_preds) / 2
            z0s = self.compute_tweedie(
                    zts, combined_delta_preds, timestep, alphas, sigmas, **kwargs
            )
            z_t_1 = self.compute_prev_state(
                    zts, z0s, timestep, combined_delta_preds, **kwargs
            )

            out_params = {
                "x0s": None,
                "z0s": z0s,
                "x_t_1": x_t_1,
                "z_t_1": z_t_1,
            }
            return out_params

        elif case_name == "58":
            # canonical
            zts = input_params["zts"]
            xts_from_zts = self.forward_mapping(zts, **kwargs)
            canonical_eps_preds = self.compute_noise_preds(xts_from_zts, timestep, **kwargs)
            canonical_delta_preds = self.inverse_mapping(canonical_eps_preds, var_type="eps", **kwargs) # for z_t_1 computation.
            canonical_x0s = self.compute_tweedie(
                xts_from_zts, canonical_eps_preds, timestep, alphas, sigmas, **kwargs
            )
            canonical_z0s = self.inverse_mapping(canonical_x0s, var_type="tweedie", **kwargs)

            # instance
            xts = input_params["xts"]
            instance_eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            instance_x0s = self.compute_tweedie(
                xts, instance_eps_preds, timestep, alphas, sigmas, **kwargs
            )
            instance_z0s = self.inverse_mapping(instance_x0s, var_type="tweedie", **kwargs)

            # combined
            combined_z0s = (canonical_z0s + instance_z0s) / 2
            z_t_1 = self.compute_prev_state(
                zts, combined_z0s, timestep, canonical_delta_preds, **kwargs
            ) # TODO: canonical_delta_preds would be wrong in DPDM sampling.
            
            # instance denoising
            x_t_1 = self.compute_prev_state(
                xts, instance_x0s, timestep, instance_eps_preds, **kwargs
            )

            out_params = {
                "x0s": None,
                "z0s": combined_z0s,
                "x_t_1": x_t_1,
                "z_t_1": z_t_1,
            }
            return out_params

        elif case_name == "59":
            # canonical
            zts = input_params["zts"]
            xts_from_zts = self.forward_mapping(zts, **kwargs)
            canonical_eps_preds = self.compute_noise_preds(xts_from_zts, timestep, **kwargs)
            canonical_delta_preds = self.inverse_mapping(
                canonical_eps_preds, var_type="eps", **kwargs
            ) # for canonical_z_t_1 computation

            canonical_x0s = self.compute_tweedie(
                xts_from_zts, canonical_eps_preds, timestep, alphas, sigmas, **kwargs
            )
            canonical_x_t_1 = self.compute_prev_state(
                xts_from_zts, canonical_x0s, timestep, canonical_delta_preds, **kwargs
            )
            canonical_z_t_1 = self.inverse_mapping(
                canonical_x_t_1, var_type="latent", **kwargs
            )

            # for visualization #
            canonical_z0s = self.inverse_mapping(
                canonical_x0s, var_type="tweedie", **kwargs
            )

            # instance
            xts = input_params["xts"]
            instance_eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
            instance_x0s = self.compute_tweedie(
                xts, instance_eps_preds, timestep, alphas, sigmas, **kwargs
            )
            instance_x_t_1 = self.compute_prev_state(
                xts, instance_x0s, timestep, instance_eps_preds, **kwargs
            )
            instance_z_t_1 = self.inverse_mapping(instance_x_t_1, var_type="latent", **kwargs)

            # combined
            combined_z_t_1 = (canonical_z_t_1 + instance_z_t_1) / 2
            
            out_params = {
                "x0s": None,
                "z0s": combined_z_t_1, # dummy
                "x_t_1": instance_x_t_1,
                "z_t_1": combined_z_t_1,
            }
            return out_params
        else:
            raise NotImplementedError(f"{case_name}")

