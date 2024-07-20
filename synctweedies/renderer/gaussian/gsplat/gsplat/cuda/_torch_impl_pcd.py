import struct
from typing import Optional, Tuple, Any

import torch
import torch.nn.functional as F
from torch import Tensor


def _persp_proj_pcd(
    means: Tensor,  # [C, N, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of prespective projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        Ks: Camera intrinsics. [C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
    """

    tz = means[..., 2]  # [C, N]


    means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means)  # [C, N, 2]
    means2d = means2d / tz[..., None]  # [C, N, 2]
    return means2d  # [C, N, 2]


def _world_to_cam_pcd(
    means: Tensor,  # [N, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of world to camera transformation on Gaussians.

    Args:
        means: Gaussian means in world coordinate system. [C, N, 3].
        viewmats: world to camera transformation matrices. [C, 4, 4].

    Returns:
        A tuple:

        - **means_c**: Gaussian means in camera coordinate system. [C, N, 3].
    """
    R = viewmats[:, :3, :3]  # [C, 3, 3]
    t = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]  # (C, N, 3)
    return means_c


def _fully_fused_projection_pcd(
    means: Tensor,  # [N, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    means_c = _world_to_cam_pcd(means, viewmats)
    means2d = _persp_proj_pcd(means_c, Ks, width, height)


    depths = means_c[..., 2]  # [C, N]

    radius = torch.ones_like(depths)  # [C, N]

    valid = (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius > 0)
        & (means2d[..., 0] - radius < width)
        & (means2d[..., 1] + radius > 0)
        & (means2d[..., 1] - radius < height)
    )
    radius[~inside] = 0.0

    radii = radius.int()
    return radii, means2d, depths


@torch.no_grad()
def _isect_tiles_pcd(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pytorch implementation of `gsplat.cuda._wrapper.isect_tiles()`.

    .. note::

        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    C, N = means2d.shape[:2]
    device = means2d.device

    # compute tiles_per_gauss
    # tile_means2d = means2d
    # tile_radii = radii
    # tile_mins = torch.floor(tile_means2d - tile_radii[..., None]).int()
    # tile_maxs = torch.ceil(tile_means2d + tile_radii[..., None]).int()
    # tile_mins[..., 0] = torch.clamp(tile_mins[..., 0], 0, tile_width)
    # tile_mins[..., 1] = torch.clamp(tile_mins[..., 1], 0, tile_height)
    # tile_maxs[..., 0] = torch.clamp(tile_maxs[..., 0], 0, tile_width)
    # tile_maxs[..., 1] = torch.clamp(tile_maxs[..., 1], 0, tile_height)
    # tiles_per_gauss = (tile_maxs - tile_mins).prod(dim=-1)  # [C, N]
    # tiles_per_gauss *= radii > 0.0

    # n_isects = tiles_per_gauss.sum().item()
    # isect_ids = torch.empty(n_isects, dtype=torch.int64, device=device)
    # flatten_ids = torch.empty(n_isects, dtype=torch.int32, device=device)

    tile_pos = torch.floor(means2d).int()
    valid_mask = radii > 0.0
    tiles_per_gauss = torch.zeros_like(radii, dtype=torch.int32)
    tiles_per_gauss[valid_mask] = 1
    n_valid = (radii > 0.0).sum().item()
    isect_ids = torch.empty(n_valid, dtype=torch.int64, device=device)
    flatten_ids = torch.empty(n_valid, dtype=torch.int32, device=device)

    cum_tiles_per_gauss = torch.cumsum(tiles_per_gauss.flatten(), dim=0)
    tile_n_bits = (tile_width * tile_height).bit_length()

    def binary(num):
        return "".join("{:0>8b}".format(c) for c in struct.pack("!f", num))

    def kernel(cam_id, gauss_id):
        if radii[cam_id, gauss_id] <= 0.0:
            return
        index = cam_id * N + gauss_id
        curr_idx = cum_tiles_per_gauss[index - 1] if index > 0 else 0

        depth_id = struct.unpack("i", struct.pack("f", depths[cam_id, gauss_id]))[0]
        x = tile_pos[cam_id, gauss_id, 0].item()
        y = tile_pos[cam_id, gauss_id, 1].item()
        tile_id = y * tile_width + x
        isect_ids[curr_idx] = (
            (cam_id << 32 << tile_n_bits) | (tile_id << 32) | depth_id
        )
        print('a',(isect_ids[curr_idx])>>32)
        print('b',((cam_id << 32 << tile_n_bits) | (tile_id << 32) | depth_id)>>32)
        flatten_ids[curr_idx] = index  # flattened index

    for cam_id in range(C):
        for gauss_id in range(N):
            kernel(cam_id, gauss_id)

    if sort:
        isect_ids, sort_indices = torch.sort(isect_ids)
        flatten_ids = flatten_ids[sort_indices]

    return tiles_per_gauss.int(), isect_ids, flatten_ids


@torch.no_grad()
def _isect_offset_encode(
    isect_ids: Tensor, C: int, tile_width: int, tile_height: int
) -> Tensor:
    """Pytorch implementation of `gsplat.cuda._wrapper.isect_offset_encode()`.

    .. note::

        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    tile_n_bits = (tile_width * tile_height).bit_length()
    tile_counts = torch.zeros(
        (C, tile_height, tile_width), dtype=torch.int64, device=isect_ids.device
    )

    isect_ids_uq, counts = torch.unique_consecutive(isect_ids >> 32, return_counts=True)

    cam_ids_uq = isect_ids_uq >> tile_n_bits
    tile_ids_uq = isect_ids_uq & ((1 << tile_n_bits) - 1)
    tile_ids_x_uq = tile_ids_uq % tile_width
    tile_ids_y_uq = tile_ids_uq // tile_width

    tile_counts[cam_ids_uq, tile_ids_y_uq, tile_ids_x_uq] = counts

    cum_tile_counts = torch.cumsum(tile_counts.flatten(), dim=0).reshape_as(tile_counts)
    offsets = cum_tile_counts - tile_counts
    return offsets.int()


def _eval_sh_bases_fast(basis_dim: int, dirs: Tensor):
    """
    Evaluate spherical harmonics bases at unit direction for high orders
    using approach described by
    Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/


    :param basis_dim: int SH basis dim. Currently, only 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)

    See reference C++ code in https://jcgt.org/published/0002/02/06/code.zip
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )

    result[..., 0] = 0.2820947917738781

    if basis_dim <= 1:
        return result

    x, y, z = dirs.unbind(-1)

    fTmpA = -0.48860251190292
    result[..., 2] = -fTmpA * z
    result[..., 3] = fTmpA * x
    result[..., 1] = fTmpA * y

    if basis_dim <= 4:
        return result

    z2 = z * z
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    result[..., 7] = fTmpB * x
    result[..., 5] = fTmpB * y
    result[..., 8] = fTmpA * fC1
    result[..., 4] = fTmpA * fS1

    if basis_dim <= 9:
        return result

    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    result[..., 13] = fTmpC * x
    result[..., 11] = fTmpC * y
    result[..., 14] = fTmpB * fC1
    result[..., 10] = fTmpB * fS1
    result[..., 15] = fTmpA * fC2
    result[..., 9] = fTmpA * fS2

    if basis_dim <= 16:
        return result

    fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
    fTmpC = 3.31161143515146 * z2 - 0.47308734787878
    fTmpB = -1.770130769779931 * z
    fTmpA = 0.6258357354491763
    fC3 = x * fC2 - y * fS2
    fS3 = x * fS2 + y * fC2
    result[..., 20] = 1.984313483298443 * z2 * (
        1.865881662950577 * z2 - 1.119528997770346
    ) + -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    result[..., 21] = fTmpD * x
    result[..., 19] = fTmpD * y
    result[..., 22] = fTmpC * fC1
    result[..., 18] = fTmpC * fS1
    result[..., 23] = fTmpB * fC2
    result[..., 17] = fTmpB * fS2
    result[..., 24] = fTmpA * fC3
    result[..., 16] = fTmpA * fS3
    return result


def _spherical_harmonics(
    degree: int,
    dirs: torch.Tensor,  # [..., 3]
    coeffs: torch.Tensor,  # [..., K, 3]
):
    """Pytorch implementation of `gsplat.cuda._wrapper.spherical_harmonics()`."""
    dirs = F.normalize(dirs, p=2, dim=-1)
    num_bases = (degree + 1) ** 2
    bases = torch.zeros_like(coeffs[..., 0])
    bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)
    return (bases[..., None] * coeffs).sum(dim=-2)
