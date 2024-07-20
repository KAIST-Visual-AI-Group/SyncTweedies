from PIL import Image
from math import pi, sqrt
import torch
from torchvision.transforms import Resize, InterpolationMode
from collections import defaultdict

from synctweedies.renderer.gaussian.gaussian import render, GSModel
from synctweedies.renderer.gaussian.gsplat.gsplat import quat_scale_to_covar_preci
import numpy as np
import torch
from torch.nn import functional as F

from diffusers.models.attention_processor import Attention


# A fast decoding method based on linear projection of latents to rgb
@torch.no_grad()
def latent_preview(x):
    # adapted from https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7
    v1_4_latent_rgb_factors = torch.tensor(
        [
            #   R        G        B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ],
        dtype=x.dtype,
        device=x.device,
    )
    image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.float()
    image = image.cpu()
    image = image.numpy()
    return image


@torch.no_grad()
def decode_latents(vae, latents):

    latents = 1 / vae.config.scaling_factor * latents

    imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)

    return imgs


@torch.no_grad()
def encode_imgs(vae, imgs):
    # imgs: [B, 3, H, W]

    imgs = 2 * imgs - 1

    posterior = vae.encode(imgs).latent_dist
    latents = posterior.sample() * vae.config.scaling_factor

    return latents


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


## Consistent noise generator


def fibonacci_sphere(N, R):
    # Generate a sequence of indices
    indices = torch.arange(0, N, dtype=torch.float32, device="cuda")

    # Golden angle in radians
    phi = pi * (3.0 - sqrt(5.0))  # Approximately 2.39996323

    # Calculate the y coordinates of points along the vertical axis
    y = 1 - (indices / (N - 1)) * 2  # y goes from 1 to -1
    radius = torch.sqrt(1 - y * y)  # Radius of the horizontal circle at height y

    # Calculate the azimuthal angle theta
    theta = phi * indices

    # Cartesian coordinates
    x = torch.cos(theta) * radius * R
    z = torch.sin(theta) * radius * R
    y *= R

    # Stack the coordinates
    points = torch.stack((x, y, z), dim=1)

    return points


def prepare_consistent_noise(
    model, cameras, resample=False, resample_scale=1.0, op_fg=0.3, op_bg=7.5
):
    # Clone model
    model = model.clone()

    # Sample gaussian noise for each gaussian
    N = len(model)

    # Resample pointcloud
    new_pcd = []
    # new_opacities = []

    if resample:
        fg_gs_num = int(N * resample_scale)
        means = model.xyz
        quats = model.quats / torch.norm(model.quats, dim=1, keepdim=True)
        scales = torch.exp(model.scales)
        volumes = scales.norm(dim=1) ** 2 * torch.sigmoid(model.opacities)
        covs = quat_scale_to_covar_preci(quats, scales, True, False)[0]
        probs = volumes / volumes.sum()
        sampled_idx = torch.multinomial(probs, fg_gs_num, replacement=True)
        counts = torch.bincount(sampled_idx)
        counts = torch.cat(
            [
                counts,
                torch.zeros(N - len(counts), device=counts.device, dtype=counts.dtype),
            ]
        )
        cnt_unique = torch.unique(counts)

        low_pass_filter = torch.eye(3, device="cuda") * 1e-8
        for n in cnt_unique:
            if n == 0:
                continue
            partial_means = means[counts == n]
            partial_covs = covs[counts == n]
            partial_opacities = model.opacities[counts == n]
            samples = (
                torch.distributions.MultivariateNormal(
                    partial_means, partial_covs + low_pass_filter
                )
                .sample((n.item(),))
                .reshape(-1, 3)
            )
            new_pcd.append(samples)
            # new_opacities.append(partial_opacities.repeat(n))
    else:
        fg_gs_num = int(N * resample_scale)
        new_pcd.append(model.xyz)
        # new_opacities.append(model.opacities)

    # add 'background' gaussians
    bg_gs_num = 8_000_000
    bg_gs = fibonacci_sphere(bg_gs_num, 10.0).to("cuda")

    new_pcd.append(bg_gs)
    # new_opacities.append(torch.full((M,), -3.0, device="cuda"))

    new_pcd = torch.cat(new_pcd, dim=0)
    # new_opacities = torch.cat(new_opacities, dim=0)
    new_opacities = torch.empty(len(new_pcd), device="cuda")

    avg_gpp = fg_gs_num / (cameras.width * cameras.height)
    opacity_dump = max(1 - (0.01) ** (op_fg / avg_gpp), 0.0)  # , 0.0001)
    new_opacities[:fg_gs_num] = torch.logit(torch.Tensor([opacity_dump]))

    avg_gpp = bg_gs_num / (cameras.width * cameras.height)
    opacity_dump = max(1 - (0.01) ** (op_bg / avg_gpp), 0.0)  # , 0.0001)
    new_opacities[fg_gs_num:] = torch.logit(torch.Tensor([opacity_dump]))

    model.xyz = new_pcd
    model.opacities = new_opacities

    return model


@torch.no_grad()
def get_consistent_noise(model, camera_set, channel=4):
    new_features = torch.randn(len(model), channel, device=model.device)
    render_pkg = render(
        model,
        camera_set,
        render_mode="PCD",
        sqrt_mode=True,
        override_color=new_features,
    )
    image = render_pkg["image"]
    transmittance = 1.0 - render_pkg["alpha"]
    pure_noise = torch.randn_like(image)
    image = image + pure_noise * torch.sqrt(transmittance)
    return image
