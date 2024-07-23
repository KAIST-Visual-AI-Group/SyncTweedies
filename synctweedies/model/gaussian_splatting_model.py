import os
import json
from tqdm import trange
import numpy as np
import torch
from typing import Optional, Literal
import torch.nn.functional as F
from diffusers import ControlNetModel
from diffusers.models import ControlNetModel
from diffusers.utils import is_compiled_module
from torchvision.transforms import Compose, GaussianBlur, InterpolationMode, Resize

from synctweedies.utils.gaussian_utils import l1_loss
from synctweedies.utils.image_utils import *
from synctweedies.model.base_model import BaseModel

from synctweedies.method_configs.case_config import (
    CANONICAL_DENOISING_ZT,
    INSTANCE_DENOISING_XT,
    JOINT_DENOISING_XT,
    JOINT_DENOISING_ZT,
    METHOD_MAP,
)
from synctweedies.utils.gaussian_utils import *
from synctweedies.renderer.gaussian.gaussian import render, GSModel
from synctweedies.utils.mesh_utils import split_groups
from synctweedies.renderer.gaussian.dataset_loader import load_cameras
from synctweedies.utils.gaussian_utils import (
    prepare_consistent_noise,
    get_consistent_noise,
)


color_constants = {
    "black": [-1, -1, -1],
    "white": [1, 1, 1],
    "maroon": [0, -1, -1],
    "red": [1, -1, -1],
    "olive": [0, 0, -1],
    "yellow": [1, 1, -1],
    "green": [-1, 0, -1],
    "lime": [-1, 1, -1],
    "teal": [-1, 0, 0],
    "aqua": [-1, 1, 1],
    "navy": [-1, -1, 0],
    "blue": [-1, -1, 1],
    "purple": [0, -1, 0],
    "fuchsia": [1, -1, 1],
}
color_names = list(color_constants.keys())


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


@torch.no_grad()
def decode_normalized_depth(depths, mask, batched_norm=False):
    # view_z, mask = depths.unbind(-1)
    view_z = depths * mask + (torch.max(depths) * (1 - mask))
    inv_z = 1 / view_z
    inv_z_min = inv_z * mask + (torch.max(depths) * (1 - mask))
    if not batched_norm:
        max_ = torch.max(inv_z, 1, keepdim=True)
        max_ = torch.max(max_[0], 2, keepdim=True)[0]

        min_ = torch.min(inv_z_min, 1, keepdim=True)
        min_ = torch.min(min_[0], 2, keepdim=True)[0]
    else:
        max_ = torch.max(inv_z)
        min_ = torch.min(inv_z_min)
    inv_z = (inv_z - min_) / (max_ - min_)
    inv_z = inv_z.clamp(0, 1)
    inv_z = inv_z[..., None].repeat(1, 1, 1, 3)

    return inv_z


class GaussianSplattingModel(BaseModel):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{self.config.gpu}")
        
        self.initialize()
        super().__init__()
        

    def initialize(self):
        super().initialize()
        self.generator = torch.manual_seed(self.config.seed)
        
        self.result_dir = f"{self.output_dir}/results"
        self.intermediate_dir = f"{self.output_dir}/intermediate"
        self.gaussian_dir = f"{self.output_dir}/gaussian"
        self.debug_dir = f"{self.output_dir}/debug"

        dirs = [
            self.intermediate_dir,
            self.result_dir,
            self.gaussian_dir,
            self.debug_dir,
        ]
        for dir_ in dirs:
            if not os.path.isdir(dir_):
                os.makedirs(dir_, exist_ok=True)

        self.logging_config = {
            "output_dir": self.output_dir,
            "log_interval": self.config.log_interval,
        }

        log_opt = vars(self.config)
        config_path = os.path.join(self.output_dir, "gs_run_config.yaml")
        with open(config_path, "w") as f:
            json.dump(log_opt, f, indent=4)

    def init_mapper(self):
        self.gaussians = GSModel(
            {
                "plyfile": self.config.plyfile,
                "xyz_lr": self.config.xyz_lr,
                "scale_lr": self.config.scale_lr,
                "quat_lr": self.config.quat_lr,
                "opacity_lr": self.config.opacity_lr,
                "feature_lr": self.config.feature_lr,
            }
        )
        torch.zero_(self.gaussians.features)

        xyz = self.gaussians.xyz
        xyz_mean = torch.mean(xyz, dim=0)
        dists = torch.linalg.norm(xyz - xyz_mean, dim=1)
        xyz_top_1_percent = torch.quantile(dists, 0.95)  # 5% quantile
        cam_dist = xyz_top_1_percent * 2.0

        self.eval_camera_poses = None
        self.rgb_camera_poses = load_cameras(
            self.config.dataset_type,
            size=[self.config.rgb_view_size, self.config.rgb_view_size],
            source_path=self.config.source_path,
            up_vec=self.config.up_vec,
            dist=cam_dist,
        ).to("cuda")
        self.latent_camera_poses = load_cameras(
            self.config.dataset_type,
            size=[self.config.latent_view_size, self.config.latent_view_size],
            source_path=self.config.source_path,
            up_vec=self.config.up_vec,
            dist=cam_dist,
        ).to("cuda")

        if len(self.rgb_camera_poses) > self.config.total_image_to_use:
            A = len(self.rgb_camera_poses)
            B = self.config.total_image_to_use
            idxs = [(i * A) // B for i in range(B)]
            non_idxs = [i for i in range(A) if i not in idxs]
            assert len(set(idxs)) == len(
                idxs
            ), f"Redundant cameras in rgb_camera_poses. It seems a bug of the code. {len(set(idxs))} != {len(idxs)}"

            self.eval_camera_poses = self.rgb_camera_poses[non_idxs]
            self.rgb_camera_poses = self.rgb_camera_poses[idxs]
            self.latent_camera_poses = self.latent_camera_poses[idxs]

        self.rasterize_mode = "antialiased" if self.config.antialiased else "classic"

        self.attention_mask = []
        if self.config.dataset_type == "sparse":
            print("Using hard-coded attention mask")
            self.attention_mask = [
                [7, 0, 1],
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
                [6, 7, 0],
                [4, 8],
                [0, 9],
            ]
            ref_views = [4]
        else:
            cam_count = len(self.latent_camera_poses)
            for i in range(len(self.latent_camera_poses)):
                self.attention_mask.append(
                    [(cam_count + i - 1) % cam_count, i, (i + 1) % cam_count]
                )
            ref_views = []
            if len(ref_views) == 0:
                ref_views = [cam_count // 2]

        self.group_metas = split_groups(
            self.attention_mask, self.config.max_batch_size, ref_views
        )

        color_images = (
            torch.FloatTensor([color_constants[name] for name in color_names])
            .reshape(-1, 3, 1, 1)
            .to(
                dtype=self.model.text_encoder.dtype, device=self.model._execution_device
            )
        )
        color_images = (
            torch.ones(
                (
                    1,
                    1,
                    self.config.latent_view_size * 8,
                    self.config.latent_view_size * 8,
                ),
                device=self.model._execution_device,
                dtype=torch.float32,
            )
            * color_images
        )
        color_images *= (0.5 * color_images) + 0.5
        color_latents = encode_imgs(
            self.model.vae, color_images.type(self.model.vae.dtype)
        )

        self.color_images = {
            color[0]: color[1]
            for color in zip(color_names, [img for img in color_images])
        }
        self.color_latents = {
            color[0]: color[1]
            for color in zip(color_names, [latent for latent in color_latents])
        }

        bg_color = 1 if self.config.background_color == "white" else 0
        self.BACKGROUND = torch.full((3,), bg_color, dtype=torch.float32, device="cuda")
        self.BACKGROUND_RGB = torch.full((1, 3, 512, 512), bg_color, device="cuda")
        with torch.no_grad():
            self.BACKGROUND_LATENT = encode_imgs(
                self.model.vae, self.BACKGROUND_RGB.half()
            )
            assert self.BACKGROUND_LATENT.shape == (
                1,
                4,
                64,
                64,
            ), self.BACKGROUND_LATENT.shape

        normals_transforms = Compose(
            [
                Resize(
                    (self.config.rgb_view_size, self.config.rgb_view_size),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                GaussianBlur(5, 5 // 3 + 1),
            ]
        )

        rendering = render(
            self.gaussians,
            self.rgb_camera_poses,
            rasterize_mode=self.rasterize_mode,
            render_mode="ED",
        )
        mask = (rendering["alpha"].squeeze(1) > 0.01).type(torch.float32)  # B H W
        depth = rendering["image"].squeeze(1) * mask + 15.0 * (1 - mask)  # B H W
        assert depth.dim() == 3, depth.shape
        norm_per_depth = decode_normalized_depth(depth, mask, True).permute(0, 3, 1, 2)
        processed_per_depth = normals_transforms(norm_per_depth)
        self.conditioning_images = processed_per_depth.to(dtype=self.model.vae.dtype)
        per_view_mask = Resize(self.config.rgb_view_size, antialias=True)(
            mask
        ).unsqueeze(1)
        per_view_mask = torch.cat([per_view_mask] * 3, dim=1)  # B 3 H W

        assert (
            self.conditioning_images.dim() == 4
            and self.conditioning_images.shape[1] == 3
        ), self.conditioning_images.shape
        save_tensor(
            self.conditioning_images,
            f"{self.intermediate_dir}/cond.png",
            save_type="cat_image",
        )

        rendering = render(
            self.gaussians,
            self.rgb_camera_poses,
            rasterize_mode=self.rasterize_mode,
            render_mode="D",
        )
        self.rgb_masks = (rendering["alpha"] > 0.01).type(torch.float32)  # B 1 H W

        rendering = render(
            self.gaussians,
            self.latent_camera_poses,
            rasterize_mode=self.rasterize_mode,
            render_mode="D",
        )
        self.latent_masks = (rendering["alpha"] > 0.01).type(torch.float32)  # B 1 H W

        assert (
            self.rgb_masks.dim() == self.latent_masks.dim() == 4
        ), f"{self.rgb_masks.dim()}, {self.latent_masks.dim()}"
        save_tensor(
            self.rgb_masks,
            f"{self.intermediate_dir}/rgb_mask.png",
            save_type="cat_image",
            is_grayscale=True,
        )
        save_tensor(
            self.latent_masks,
            f"{self.intermediate_dir}/latent_mask.png",
            save_type="cat_image",
            is_grayscale=True,
        )

        self.canonical_domain = self.config.canonical_domain
        if self.config.case_num != "2":
            self.canonical_domain = "latent"

        self.channels = 3 if self.canonical_domain == "rgb" else 4

        self.previous_features = torch.zeros(
            (len(self.gaussians), self.channels),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )

        print("Done Initialization")
        pass

    @torch.enable_grad()
    def init_noise(self, noise_type, **kwargs):
        device = self.model.vae.device
        dtype = self.model.vae.dtype
        if noise_type == "random":
            eps = torch.randn(
                (
                    len(self.rgb_camera_poses),
                    4,
                    self.config.latent_view_size,
                    self.config.latent_view_size,
                ),
                generator=self.generator,
                dtype=dtype,
            ).to(device)
        elif noise_type == "consistent":
            tmp_model = prepare_consistent_noise(
                self.gaussians, self.latent_camera_poses
            )
            eps = get_consistent_noise(tmp_model, self.latent_camera_poses).to(
                dtype=dtype, device=device
            )
        else:
            raise NotImplementedError(f"{noise_type}")

        return eps

    @torch.no_grad()
    def forward_mapping(
        self,
        z_t: torch.Tensor,
        canonical_type: Literal["rgb", "latent"],
        instance_type: Literal["rgb", "latent"],
        bg: Optional[torch.Tensor] = None,
        cur_index: int = -1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Mapping canonical to instance domain by rendering gaussians.

        Args:
            z_t (torch.Tensor): The canonical data tensor.
            canonical_type (str): One of ('rgb', 'latent'). Specifies the type of canonical data.
            output_type (str): One of ('rgb', 'latent'). Specifies the desired output type.
            bg (torch.Tensor, optional): Background tensor. Defaults to None.
            t (int, optional): Timestep value for noising. Defaults to 0.

        Returns:
            torch.Tensor: The mapped tensor in the desired output type.

        Raises:
            ValueError: If either canonical_type or output_type is not one of ('rgb', 'latent').
        """

        if canonical_type == "rgb":
            camera_poses = self.rgb_camera_poses
            override_color = F.sigmoid(z_t)
        else:
            camera_poses = self.latent_camera_poses
            override_color = z_t

        rendering = render(
            self.gaussians,
            camera_poses,
            rasterize_mode=self.rasterize_mode,
            override_color=override_color,
        )
        images = rendering["image"]
        alphas = rendering["alpha"]

        if canonical_type == "rgb":
            assert (
                self.config.force_clean_composition or bg != None or cur_index == -1
            ), "Do not support noisy background in canonical rgb"
            if self.config.force_clean_composition or bg == None:
                bg = self.BACKGROUND_RGB

            if bg.shape[1] == 3:
                images = images * alphas + bg * (1 - alphas)
            if instance_type == "latent":
                with torch.no_grad():
                    images = encode_imgs(self.model.vae, images.half())
                if bg.shape[1] == 4:
                    images = images * self.latent_masks + bg * (1 - self.latent_masks)
        elif canonical_type == "latent":
            if self.config.force_clean_composition:
                bg = self.BACKGROUND_LATENT
            elif bg == None:
                t = self.model.scheduler.timesteps[cur_index]
                if t != 0:
                    bg = self.model.scheduler.add_noise(
                        self.BACKGROUND_LATENT,
                        torch.randn_like(self.BACKGROUND_LATENT),
                        t,
                    )
                else:
                    bg = self.BACKGROUND_LATENT

            if bg.shape[1] == 4:
                images = images * alphas + bg * (1 - alphas)
            if instance_type == "rgb":
                with torch.no_grad():
                    images = decode_latents(images.half(), self.model.vae)
                if bg.shape[1] == 3:
                    images = images * self.rgb_masks + bg * (1 - self.rgb_masks)

        assert images.dim() == 4, images.shape
        return images

    @torch.enable_grad()
    def inverse_mapping(
        self,
        x_t: torch.Tensor,
        instance_type: Literal["rgb", "latent"],
        canonical_type: Literal["rgb", "latent"],
        iterations: Optional[int] = None,
        t: int = 0,
        color_assign_method: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Inverse mapping from instance to canonical domain.

        Args:
            x_t (torch.Tensor): The instance data tensor.
            instance_type (str): The type of instance data, must be one of ('rgb', 'latent').
            canonical_type (str): The type of canonical data, must be one of ('rgb', 'latent').
            iterations (int, optional): Number of iterations for the inverse mapping process. Defaults to None.
            t (int, optional): Timestep value. Defaults to 0.

        Returns:
            torch.Tensor: The tensor in the canonical domain.

        Raises:
            ValueError: If instance_type or canonical_type is not one of ('rgb', 'latent').
        """

        # Set GT images and other variables
        pred_original_sample = x_t
        if canonical_type == "rgb":
            camera_poses = self.rgb_camera_poses
            bg = self.BACKGROUND_RGB
            if instance_type == "latent":
                with torch.no_grad():
                    pred_original_sample = decode_latents(
                        self.model.vae, pred_original_sample.half()
                    )
                    pred_original_sample = torch.clamp(
                        pred_original_sample, 0.001, 0.999
                    )  # clip between 0.001 and 0.999
        elif canonical_type == "latent":
            camera_poses = self.latent_camera_poses
            bg = self.BACKGROUND_LATENT
            if instance_type == "rgb":
                with torch.no_grad():
                    pred_original_sample = encode_imgs(
                        self.model.vae, pred_original_sample.half()
                    )

        # Initialize Gaussian features
        channels = 3 if canonical_type == "rgb" else 4
        color_assign_method = color_assign_method or self.config.color_assign_method

        if color_assign_method == "zero":
            features = torch.zeros(
                (len(self.gaussians), channels),
                device="cuda",
                dtype=torch.float32,
            )
        elif color_assign_method == "previous":
            if channels == self.previous_features.shape[1]:
                features = self.previous_features
            else:
                print(
                    "Previous features are not same as current features. Zeroing out."
                )
                features = torch.zeros(
                    (len(self.gaussians), channels),
                    device="cuda",
                    dtype=torch.float32,
                )
        else:
            raise NotImplementedError(f"{color_assign_method}")

        features = features.detach()
        features.requires_grad = True

        self.gaussians.features = features
        parameters = ["features"]
        if self.config.enable_opacity:
            parameters.append("opacities")
        if self.config.enable_xyz:
            parameters.append("xyz")
        if self.config.enable_covariance:
            parameters.append("scales")
            parameters.append("quats")
        self.gaussians.prepare_optimizer(parameters)

        # Do optimization
        camera_num = len(camera_poses)
        random_list = None

        iterations = iterations if iterations is not None else self.config.iterations
        for it in trange(iterations):
            if it % camera_num == 0:
                random_list = np.random.permutation(camera_num)
            idx = random_list[it % camera_num]
            viewpoint_cam = camera_poses[idx]
            gt_latent = pred_original_sample[idx].cuda()

            if canonical_type == "rgb":
                override_color = F.sigmoid(features)
            elif canonical_type == "latent":
                override_color = features

            render_pkg = render(
                self.gaussians,
                viewpoint_cam,
                override_color=override_color,
                rasterize_mode=self.rasterize_mode,
                render_mode="RGB",
            )

            image = ((render_pkg["image"] + bg * (1 - render_pkg["alpha"])))[0].type(
                gt_latent.dtype
            )

            # Masked loss
            image = image
            gt_latent_composed = gt_latent
            loss = l1_loss(image, gt_latent_composed)
            loss.backward()
            self.gaussians.step()
            self.gaussians.zero_grad()

        self.previous_features = features.detach()
        return features.clone().detach()

    def save_output(self):
        """
        Pass dictionary for saving outputs with keys as filenames
        """
        pass

    def initialize_latent(self):
        pass

    def compute_noise_preds(self, xts, timestep, **kwargs):

        return self.model.compute_noise_preds(xts, timestep, **kwargs)

    @torch.no_grad()
    def __call__(self):
        if os.path.exists(os.path.join(self.gaussian_dir, "final_model.ply")):
            print("Gaussian model already exists! Overwriting...")

        num_timesteps = self.model.scheduler.config.num_train_timesteps
        initial_controlnet_conditioning_scale = self.config.conditioning_scale
        controlnet_conditioning_scale = self.config.conditioning_scale
        controlnet_conditioning_end_scale = self.config.conditioning_scale_end
        control_guidance_end = [self.config.control_guidance_end]
        control_guidance_start = [self.config.control_guidance_start]
        log_interval = self.config.log_interval
        ref_attention_end = self.config.ref_attention_end
        multiview_diffusion_end = self.config.mvd_end
        prompt = f"Best quality, extremely detailed {self.config.prompt}"
        negative_prompt = self.config.negative_prompt

        controlnet = (
            self.model.controlnet._orig_mod
            if is_compiled_module(self.model.controlnet)
            else self.model.controlnet
        )

        # 1. Prepare some variables
        device = self.model._execution_device
        guidance_scale = self.config.guidance_scale

        num_inference_steps = self.config.steps
        self.model.scheduler.set_timesteps(self.config.steps, device=device)
        timesteps = self.model.scheduler.timesteps
        
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.model.scheduler.order
        )
        intermediate_results = []

        main_views = []
        exp = 0
        alphas = self.model.scheduler.alphas_cumprod ** (0.5)
        sigmas = (1 - self.model.scheduler.alphas_cumprod) ** (0.5)

        # 2. Prepare extra step kwargs.
        eta = 0.0
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(self.generator, eta)

        # 3. Encode input prompt
        prompt_embeds = self.model._encode_prompt(
            prompt,
            device,
            1,
            True,
            negative_prompt=negative_prompt,
        )
        neg_embeds, pos_embeds = torch.chunk(prompt_embeds, 2)
        positive_prompt_embeds = torch.cat(
            [pos_embeds] * len(self.latent_camera_poses), axis=0
        )

        negative_prompt_embeds = torch.cat(
            [neg_embeds] * len(self.latent_camera_poses), axis=0
        )

        # 4. Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(
                keeps[0] if isinstance(controlnet, ControlNetModel) else keeps
            )

        # 5. Prepare latent variables
        case_name = METHOD_MAP[str(self.config.case_num)]
        is_denoizing_zt = (
            case_name in CANONICAL_DENOISING_ZT or case_name in JOINT_DENOISING_ZT
        )

        if is_denoizing_zt:
            num_points = len(self.gaussians)
            latents = torch.randn((num_points, 4), device=device)  # (N, 4)
        else:
            noise_type = "consistent" if self.config.zt_init else "random"
            latents = self.init_noise(noise_type)
            save_tensor(
                latents[:, :3, :, :],
                f"{self.intermediate_dir}/noise.mp4",
                save_type="video",
                fps=10,
            )

        func_params = {
            "guidance_scale": guidance_scale,
            "positive_prompt_embeds": positive_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "conditioning_images": self.conditioning_images,
            "group_metas": self.group_metas,
            "controlnet_keep": controlnet_keep,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "do_classifier_free_guidance": True,
            "ref_attention_end": ref_attention_end,
            "num_timesteps": num_timesteps,
            "cross_attention_kwargs": None,
            "skip_delta_preds": True,
            "generator": self.generator,
            "predicted_variance": None,
            "exp": exp,
            "main_views": main_views,
            "cos_weighted": True,
            "canonical_type": self.canonical_domain,
            "instance_type": "latent",
        }

        # 6. Run the main loop
        with self.model.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                func_params["cur_index"] = i
                self.model.set_up_coefficients(t, self.config.sampling_method)

                if is_denoizing_zt:
                    input_params = {"zts": latents, "xts": None}
                else:
                    input_params = {"zts": None, "xts": latents}

                if t > (1 - multiview_diffusion_end) * num_timesteps:
                    res_dict = self.one_step_process(
                        input_params=input_params,
                        timestep=t,
                        alphas=alphas,
                        sigmas=sigmas,
                        case_name=case_name,
                        **func_params,
                    )

                    if is_denoizing_zt:
                        latents = res_dict["z_t_1"]
                    else:
                        latents = res_dict["x_t_1"]

                    if res_dict.get("x_t_1") is not None:
                        x_t_1 = res_dict["x_t_1"]
                    else:
                        x_t_1 = self.forward_mapping(res_dict["z_t_1"], **func_params)

                    if res_dict.get("x0s") is not None:
                        x_0 = res_dict["x0s"]
                    else:
                        x_0 = self.forward_mapping(res_dict["z0s"], **func_params)

                    intermediate_results.append((x_t_1, x_0))
                else:
                    if latents.dim() == 2:
                        latents = self.forward_mapping(latents, **func_params)

                    noise_pred = self.compute_noise_preds(latents, t, **func_params)
                    step_results = self.model.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=True
                    )

                    pred_original_sample = step_results["pred_original_sample"]
                    latents = step_results["prev_sample"]

                    intermediate_results.append(
                        (step_results["prev_sample"], pred_original_sample.to("cpu"))
                    )

                # Update pipeline settings after one step:
                # Annealing ControlNet scale
                if (1 - t / num_timesteps) < control_guidance_start[0]:
                    controlnet_conditioning_scale = (
                        initial_controlnet_conditioning_scale
                    )
                elif (1 - t / num_timesteps) > control_guidance_end[0]:
                    controlnet_conditioning_scale = controlnet_conditioning_end_scale
                else:
                    alpha = ((1 - t / num_timesteps) - control_guidance_start[0]) / (
                        control_guidance_end[0] - control_guidance_start[0]
                    )
                    controlnet_conditioning_scale = (
                        alpha * initial_controlnet_conditioning_scale
                        + (1 - alpha) * controlnet_conditioning_end_scale
                    )

                if i % log_interval == log_interval - 1 or t == 1:
                    latent_noise, latent_img = intermediate_results[-1]
                    latent_img = latent_img[0:5]

                    rgb_img = decode_latents(
                        self.model.vae,
                        latent_img.half().to(self.model._execution_device),
                    )

                    if latent_noise is not None:
                        latent_noise = latent_noise[0:5]
                        rgb_noise = decode_latents(
                            self.model.vae,
                            latent_noise.half().to(self.model._execution_device),
                        )
                        rgb_img = torch.cat((rgb_img, rgb_noise), dim=0)
                    save_tensor(
                        rgb_img,
                        f"{self.intermediate_dir}/intermediate_{i}.png",
                        save_type="cat_image",
                        row_size=2,
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.model.scheduler.order == 0
                ):
                    progress_bar.update()

        if latents.dim() == 2:
            print(
                "Converting latents to instance space. This should not happen."
            )
            latents = self.forward_mapping(
                latents,
                bg=None,
                instance_type="rgb",
                canonical_type=self.canonical_domain,
            )
        else:
            latents = decode_latents(self.model.vae, latents.half())

        assert (
            latents.dim() == 4 and latents.shape[1] == 3
        ), f"Expected 3 channels with shape (B, 3, 512, 512), got {latents.shape}"

        # 7. Final averaging
        z0s = self.inverse_mapping(
            latents,
            instance_type="rgb",
            canonical_type="rgb",
            iterations=self.config.final_iterations,
            color_assign_method="zero",
        )
        self.gaussians.features = z0s

        # 8. Save final results
        train_dir = os.path.join(
            self.logging_config["output_dir"], "train_view_results"
        )
        instance_dir = os.path.join(
            self.logging_config["output_dir"], "instance_results"
        )
        eval_dir = os.path.join(self.logging_config["output_dir"], "eval_view_results")
        train_video_name = os.path.join(
            self.logging_config["output_dir"], "train_render.mp4"
        )
        train_gif_name = os.path.join(
            self.logging_config["output_dir"], "train_render.gif"
        )
        instance_video_name = os.path.join(
            self.logging_config["output_dir"], "instance_render.mp4"
        )
        instance_gif_name = os.path.join(
            self.logging_config["output_dir"], "instance_render.gif"
        )
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(instance_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        backgrounds = self.BACKGROUND.view(1, 3)
        override_color = F.sigmoid(z0s)
        rendering = render(
            self.gaussians,
            self.rgb_camera_poses,
            rasterize_mode=self.rasterize_mode,
            override_color=override_color,
            render_mode="RGB",
            backgrounds=backgrounds.repeat(len(self.rgb_camera_poses), 1),
        )
        img_tensor = rendering["image"][:, :3]
        save_tensor(img_tensor, train_dir, save_type="images")
        save_tensor(img_tensor, train_video_name, save_type="video", fps=10)
        save_tensor(img_tensor, train_gif_name, save_type="gif", fps=10)

        if self.eval_camera_poses is not None:
            print("Rendering evaluation views...")
            override_color = F.sigmoid(z0s)
            rendering = render(
                self.gaussians,
                self.eval_camera_poses,
                rasterize_mode=self.rasterize_mode,
                override_color=override_color,
                render_mode="RGB",
                backgrounds=backgrounds.repeat(len(self.eval_camera_poses), 1),
            )
            img_tensor = rendering["image"][:, :3]
            save_tensor(img_tensor, eval_dir, save_type="images")

        save_tensor(latents, instance_dir, save_type="images")
        save_tensor(latents, instance_video_name, save_type="video", fps=10)
        save_tensor(latents, instance_gif_name, save_type="gif", fps=10)

        self.gaussians.save(os.path.join(self.gaussian_dir, "final_model.ply"))

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return None
