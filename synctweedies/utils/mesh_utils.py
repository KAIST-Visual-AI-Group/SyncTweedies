import torch
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode
import numpy as np
from collections import defaultdict

direction_names = ["", "front", "side", "back", "top", "bottom"]


# Revert time 0 background to time t to composite with time t foreground
@torch.no_grad()
# composite_rendered_view(self.scheduler, latents, foregrounds, masks, timesteps[0]+1)
def composite_rendered_view(scheduler, backgrounds, foregrounds, masks, t, add_noise):
    # backgrounds: Tensor (B 4 H W)
    # foregrounds: List (4 H W)
    # masks: List (1 H W)
    composited_images = []
    for i, (background, foreground, mask) in enumerate(zip(backgrounds, foregrounds, masks)):
        if t > 0 and add_noise:
            alphas_cumprod = scheduler.alphas_cumprod[t]
            noise = torch.normal(0, 1, background.shape, device=background.device)
            background = (1-alphas_cumprod) * noise + alphas_cumprod * background
        composited = foreground * mask + background * (1-mask)
        composited_images.append(composited)
        
    composited_tensor = torch.stack(composited_images)
    return composited_tensor


# Split into micro-batches to use less memory in each unet prediction
# But need more investigation on reducing memory usage
# Assume it has no possitive effect and use a large "max_batch_size" to skip splitting
def split_groups(attention_mask, max_batch_size, ref_view=[]):
    group_sets = []
    group = set()
    ref_group = set()
    idx = 0
    while idx < len(attention_mask):
        new_group = group | set([idx])
        new_ref_group = (ref_group | set(attention_mask[idx] + ref_view)) - new_group 
        if len(new_group) + len(new_ref_group) <= max_batch_size:
            group = new_group
            ref_group = new_ref_group
            idx += 1
        else:
            assert len(group) != 0, "Cannot fit into a group"
            group_sets.append((group, ref_group))
            group = set()
            ref_group = set()
    if len(group)>0:
        group_sets.append((group, ref_group))

    group_metas = []
    for group, ref_group in group_sets:
        in_mask = sorted(list(group | ref_group))
        out_mask = []
        group_attention_masks = []
        for idx in in_mask:
            if idx in group:
                out_mask.append(in_mask.index(idx))
            group_attention_masks.append([in_mask.index(idxx) for idxx in attention_mask[idx] if idxx in in_mask])
        ref_attention_mask = [in_mask.index(idx) for idx in ref_view]
        group_metas.append([in_mask, out_mask, group_attention_masks, ref_attention_mask])

    return group_metas


# Used to generate depth or normal conditioning images
@torch.no_grad()
def get_conditioning_images(uvp, output_size, blur_filter=5, cond_type="normal"):
    verts, normals, depths, cos_maps, texels, fragments = uvp.render_geometry(image_size=output_size)
    masks = normals[...,3][:,None,...]
    masks = Resize((output_size,)*2, antialias=True)(masks)
    normals_transforms = Compose([
        Resize((output_size,)*2, interpolation=InterpolationMode.BILINEAR, antialias=True), 
        GaussianBlur(blur_filter, blur_filter//3+1)]
    )

    if cond_type == "normal":
        view_normals = uvp.decode_view_normal(normals).permute(0,3,1,2) *2 - 1
        conditional_images = normals_transforms(view_normals)
    # Some problem here, depth controlnet don't work when depth is normalized
    # But it do generate using the unnormalized form as below
    elif cond_type == "depth":
        view_depths = uvp.decode_normalized_depth(depths).permute(0,3,1,2)
        conditional_images = normals_transforms(view_depths)
    
    return conditional_images, masks


@torch.no_grad()
def encode_latents(vae, imgs):
    imgs = (imgs-0.5)*2
    latents = vae.encode(imgs).latent_dist.sample()
    latents = vae.config.scaling_factor * latents
    return latents


@torch.no_grad()
def decode_latents(vae, latents):

    latents = 1 / vae.config.scaling_factor * latents

    image = vae.decode(latents, return_dict=False)[0]
    torch.cuda.current_stream().synchronize()

    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.permute(0, 2, 3, 1)
    image = image.float()
    image = image.cpu()
    image = image.numpy()
    
    return image

@torch.no_grad()
def decode_latents_tensor(vae, latents):

	latents = 1 / vae.config.scaling_factor * latents

	image = vae.decode(latents, return_dict=False)[0]
	torch.cuda.current_stream().synchronize()

	image = (image / 2 + 0.5).clamp(0, 1)
	
	return image


# A fast decoding method based on linear projection of latents to rgb
@torch.no_grad()
def latent_preview(x):
    # adapted from https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7
    v1_4_latent_rgb_factors = torch.tensor([
        #   R        G        B
        [0.298, 0.207, 0.208],  # L1
        [0.187, 0.286, 0.173],  # L2
        [-0.158, 0.189, 0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ], dtype=x.dtype, device=x.device)
    image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.float()
    image = image.cpu()
    image = image.numpy()
    return image


# Decode each view and bake them into a rgb texture
def get_rgb_texture(vae, uvp_rgb, latents, disable):
    result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    resize = Resize((uvp_rgb.render_size,)*2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
    result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
    textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, 
                                                                               main_views=[], 
                                                                               exp=6, 
                                                                               noisy=False,
                                                                               disable=disable)
    result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()[None,...]
    return result_tex_rgb, result_tex_rgb_output


import numpy as np
import torch
from torch.nn import functional as F

from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler, DDPMScheduler
from diffusers.models.attention_processor import Attention, AttentionProcessor


def replace_attention_processors(module, processor, attention_mask=None, ref_attention_mask=None, ref_weight=0):
    attn_processors = module.attn_processors
    for k, v in attn_processors.items():
        if "attn1" in k:
            attn_processors[k] = processor(custom_attention_mask=attention_mask, ref_attention_mask=ref_attention_mask, ref_weight=ref_weight)
    module.set_attn_processor(attn_processors)


class SamplewiseAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, custom_attention_mask=None, ref_attention_mask=None, ref_weight=0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.ref_weight = ref_weight
        self.custom_attention_mask = custom_attention_mask
        self.ref_attention_mask = ref_attention_mask

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim


        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channels = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = torch.clone(hidden_states)
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


        '''
            reshape encoder hidden state to a single batch
        '''
        encoder_hidden_states_f = encoder_hidden_states.reshape(1, -1, channels)



        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        '''
            each time select 1 sample from q and compute with concated kv
            concat result hidden states afterwards
        '''
        hidden_state_list = []

        for b_idx in range(batch_size):
            
            query_b = query[b_idx:b_idx+1]

            if self.ref_weight > 0 or True:
                key_ref = key.clone()
                value_ref = value.clone()

                keys = [key_ref[view_idx] for view_idx in self.ref_attention_mask]
                values = [value_ref[view_idx] for view_idx in self.ref_attention_mask]

                key_ref = torch.stack(keys)
                key_ref = key_ref.view(key_ref.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]

                value_ref = torch.stack(values)
                value_ref = value_ref.view(value_ref.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]

            key_a = key.clone()
            value_a = value.clone()

            # key_a = key_a[max(0,b_idx-1):min(b_idx+1,batch_size)+1]

            keys = [key_a[view_idx] for view_idx in self.custom_attention_mask[b_idx]]
            values = [value_a[view_idx] for view_idx in self.custom_attention_mask[b_idx]]

            # keys = (key_a[b_idx-1], key_a[b_idx], key_a[(b_idx+1)%batch_size])
            # values = (value_a[b_idx-1], value_a[b_idx], value_a[(b_idx+1)%batch_size])
            
            # if b_idx not in [0, batch_size-1, batch_size//2]:
            # 	keys = keys + (key_a[min(batch_size-2, 2*(batch_size//2) - b_idx)],)
            # 	values = values + (value_a[min(batch_size-2, 2*(batch_size//2) - b_idx)],)
            key_a = torch.stack(keys)
            key_a = key_a.view(key_a.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]

            # value_a = value_a[max(0,b_idx-1):min(b_idx+1,batch_size)+1]
            value_a = torch.stack(values)
            value_a = value_a.view(value_a.shape[0], -1, attn.heads, head_dim).permute(2, 0, 1, 3).contiguous().view(attn.heads, -1, head_dim)[None,...]

            hidden_state_a = F.scaled_dot_product_attention(
                query_b, key_a, value_a, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            if self.ref_weight > 0 or True:
                hidden_state_ref = F.scaled_dot_product_attention(
                    query_b, key_ref, value_ref, attn_mask=None, dropout_p=0.0, is_causal=False
                )

                hidden_state = (hidden_state_a + self.ref_weight * hidden_state_ref) / (1+self.ref_weight)
            else:
                hidden_state = hidden_state_a

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            
            hidden_state_list.append(hidden_state)

        hidden_states = torch.cat(hidden_state_list)


        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def measure_metric(pred, gt, mask=None):
    # pred: torch.Tensor B C H W [0, 1]
    # gt: torch.Tensor B C H W [0, 1]

    assert pred.shape == gt.shape

    metric = defaultdict(list)
    pred = pred.to(mask)
    if mask == None:
        mask = torch.ones_like(pred, dtype=gt.dtype, devide=gt.device)
    
    for _b in range(pred.shape[0]):
        cur_mask = (mask[_b] == 1)
        cur_pred = pred[_b]
        cur_gt = gt[_b]
        reproj_error = ((cur_pred[cur_mask] - cur_gt[cur_mask]) ** 2).mean()
        metric["reproj_error"].append(reproj_error.item())

    return metric 


def prepare_directional_prompt(prompt, negative_prompt):
    directional_prompt = [prompt + f", {v} view." for v in direction_names]
    negative_prompt = [negative_prompt + f", {v} view." for v in direction_names ]
    return directional_prompt, negative_prompt


def get_azi_elev(pose):
    rot, trans = np.transpose(pose.R), pose.T
    w2c = np.zeros((4, 4))
    w2c[:3, :3] = rot
    w2c[:3, 3] = trans
    w2c[3, 3] = 1.0
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1
    v = c2w[:3, 3]

    radius = np.linalg.norm(v)
    elev = np.arcsin(v[1] / radius)
    xz_vec = (v[0] ** 2 + v[2] ** 2) ** 0.5
    azi = np.arccos(v[2] / xz_vec)

    return np.rad2deg(elev), 180 - np.rad2deg(azi)



@torch.no_grad()
def azim_prompt(embeddings, pose, gaussian=False):
    if gaussian:
        elev, azim = get_azi_elev(pose)
    else:
        elev, azim = pose
    if elev > 30:
        pos_z = embeddings["top"]
    elif elev < -30:
        pos_z = embeddings["bottom"]
    else:
        if azim > 180:
            azim -= 360
        if azim >= -30 and azim <= 30:
            pos_z = embeddings["front"]
        elif azim <=-120 or azim >= 120:
            pos_z = embeddings["back"]
        else:
            pos_z = embeddings["side"]
    return pos_z


# Choose an opposite prompt for negative prompt
@torch.no_grad()
def azim_neg_prompt(embeddings, pose, gaussian=False):
    if gaussian:
        elev, azim = get_azi_elev(pose)
    else:
        elev, azim = pose
    if azim > 180:
        azim -= 360
    if azim > -30 and azim < 30:
        pos_z = embeddings[""]
    elif azim <=-120 or azim >= 120:
        pos_z = embeddings["front"]
    else:
        pos_z = embeddings["front"]
    return pos_z


# We can also linearly blend the prompt
# Currently not in use
@torch.no_grad()
def azim_prompt_mix(embeddings, pose):
    elev, azim = pose
    if elev >= 30:
        pos_z = embeddings["top"]
    elif elev <= -30:
        pos_z = embeddings["bottom"]
    else:
        # print(azim)
        if azim > 180:
            azim -= 360
        if azim >= -90 and azim < 90:
            if azim >= 0:
                r = 1 - azim / 90
            else:
                r = 1 + azim / 90
            start_z = embeddings['front']
            end_z = embeddings['side']
            pos_z = r * start_z + (1 - r) * end_z
        else:
            if azim >= 0:
                r = 1 - (azim - 90) / 90
            else:
                r = 1 + (azim + 90) / 90
            start_z = embeddings['side']
            end_z = embeddings['back']
            pos_z = r * start_z + (1 - r) * end_z
    return pos_z
