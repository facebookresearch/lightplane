# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.checkpoint import checkpoint

from .misc_utils import check_grid_and_color_grid, is_in_bounds, unflatten_grid
from .mlp_utils import SplatterParams, _flattened_one_mlp_params_to_list
from .naive_renderer import (
    _contract_pi,
    _depth_inv_sphere,
    _eval_mlp,
    sample_grid_list_checkpointed,
)
from .ray_utils import Rays

logger = logging.getLogger(__name__)


VERBOSE = False


if VERBOSE:
    torch.set_printoptions(
        precision=4,
        threshold=None,
        edgeitems=None,
        linewidth=120,
        profile=None,
        sci_mode=False,
    )


def lightplane_splatter_naive(
    rays: Rays,
    output_grid_size: list[tuple[int, int, int, int, int]],
    # ------ config keys ------
    num_samples: int,
    num_samples_inf: int = 0,
    mask_out_of_bounds_samples: bool = False,
    contract_coords: bool = False,
    disparity_at_inf: float = 1e-5,
    return_list: bool = True,  # whether return list or stacked tensor
    regenerate_code: bool = False,
    triton_block_size: int = 16,
    triton_num_warps: int = 4,
    checkpointing: bool = False,  # whether or not use pytorch checkpoint for MLP eval
) -> torch.Tensor | list[torch.Tensor]:
    r"""
    This is the naive implementation of the Lightplane Splatter
    (`lightplane_splatter`), which gives the same numeric results as the Triton
    implementation with less memory efficiency.
    It is useful for debugging and understanding the Triton implementation.

    Its arguments are the same as the Triton implementation in `lightplane_splatter`.
    Additionally, it could work using `torch.torch.utils.checkpoint` by setting
    `checkpointing=True`.

    Please follow the docstring of :func:`lightplane.lightplane_splatter` for
    description of arguments and returned variables.

    Note:
        The following method arguments are additional to the arguments of
        :func:`lightplane.lightplane_splatter`.

    Args:
        regenerate_code: Ignored, but kept for compatibility with triton api
        triton_block_size: Ignored, but kept for compatibility with triton api
        triton_num_warps: Ignored, but kept for compatibility with triton api
        checkpointing: Whether or not use `torch.utils.checkpoint` for checkpointing.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: The splatted results.

    References:
        [1] MERF: Memory-Efficient Radiance Fields for Real-time View Synthesis in
        Unbounded Scenes, https://arxiv.org/abs/2302.12249

    """
    return _lightplane_splatter_naive_impl(
        rays,
        output_grid_size,
        mlp_params=None,
        input_grid=None,
        # ------ config keys ------
        num_samples=num_samples,
        num_samples_inf=num_samples_inf,
        mask_out_of_bounds_samples=mask_out_of_bounds_samples,
        contract_coords=contract_coords,
        disparity_at_inf=disparity_at_inf,
        return_list=return_list,
        regenerate_code=regenerate_code,
        triton_block_size=triton_block_size,
        triton_num_warps=triton_num_warps,
        checkpointing=checkpointing,
    )


def lightplane_mlp_splatter_naive(
    rays: Rays,
    output_grid_size: list[tuple[int, int, int, int, int]],
    mlp_params: SplatterParams,
    input_grid: tuple[torch.Tensor, ...] | torch.Tensor,
    # ------ config keys ------
    num_samples: int,
    num_samples_inf: int = 0,
    mask_out_of_bounds_samples: bool = False,
    contract_coords: bool = False,
    disparity_at_inf: float = 1e-5,
    input_grid_sizes: list[list[int]] | None = None,
    return_list: bool = True,
    regenerate_code: bool = False,
    triton_block_size: int = 16,
    triton_num_warps: int = 4,
    checkpointing: bool = False,  # whether or not use pytorch checkpoint for MLP eval
) -> torch.Tensor | list[torch.Tensor]:
    r"""
    This is the naive implementation of the Lightplane Splatter with MLP and `input_grid`
    (`lightplane_mlp_splatter`), which gives the same numeric results as the Triton
    implementation with less memory efficiency.
    It is useful for debugging and understanding the Triton implementation.

    Its arguments are the same as the Triton implementation in `lightplane_mlp_splatter`.
    Additionally, it could work using `torch.torch.utils.checkpoint` by setting
    `checkpointing=True`.

    Please follow the docstring of :func:`lightplane.lightplane_mlp_splatter` for
    description of arguments and returned variables.

    Note:
        The following method arguments are additional to the arguments of
        :func:`lightplane.lightplane_mlp_splatter`.

    Args:
        regenerate_code: Ignored, but kept for compatibility with triton api
        triton_block_size: Ignored, but kept for compatibility with triton api
        triton_num_warps: Ignored, but kept for compatibility with triton api
        checkpointing: Whether or not use `torch.utils.checkpoint` for checkpointing.
    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: The splatted results.

    References:
        [1] MERF: Memory-Efficient Radiance Fields for Real-time View Synthesis in
        Unbounded Scenes, https://arxiv.org/abs/2302.12249
    """
    input_grid, _, input_grid_sizes, _ = check_grid_and_color_grid(
        input_grid, None, input_grid_sizes, None
    )

    # if grid is flatten tensor, we need to unflatten them to 5-dim tensor so that pytorch can interpolate on them.
    # unflatten use split operations, which should use no addtional memories as it creats tensor view instead of allocating memories.
    if isinstance(input_grid, torch.Tensor):
        input_grid_sizes_tensor = torch.tensor(
            input_grid_sizes, device=input_grid.device, dtype=torch.long
        )
        input_grid = unflatten_grid(input_grid, input_grid_sizes_tensor)

    return _lightplane_splatter_naive_impl(
        rays,
        output_grid_size,
        mlp_params=mlp_params,
        input_grid=input_grid,
        # ------ config keys ------
        num_samples=num_samples,
        num_samples_inf=num_samples_inf,
        mask_out_of_bounds_samples=mask_out_of_bounds_samples,
        contract_coords=contract_coords,
        disparity_at_inf=disparity_at_inf,
        input_grid_sizes=input_grid_sizes,
        return_list=return_list,
        regenerate_code=regenerate_code,
        triton_block_size=triton_block_size,
        triton_num_warps=triton_num_warps,
        checkpointing=checkpointing,
    )


def _lightplane_splatter_naive_impl(
    rays: Rays,
    output_grid_size: list[list[int]],
    mlp_params: SplatterParams | None = None,
    input_grid: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
    # ------ config keys ------
    num_samples: int = 128,
    num_samples_inf: int = 0,
    mask_out_of_bounds_samples: bool = False,
    contract_coords: bool = False,
    disparity_at_inf: float = 1e-5,
    input_grid_sizes: list[list[int]] | None = None,
    return_list: bool = True,
    regenerate_code: bool = False,
    triton_num_warps: int = -1,  # ignored, but kept for compatibility with triton api
    triton_block_size: int = -1,  # ignored, but kept for compatibility with triton api
    checkpointing: bool = False,  # whether or not use pytorch checkpoint for MLP eval
):
    device = rays.device
    if mlp_params is not None:
        use_mlp = True
        n_hidden = mlp_params.n_hidden
    else:
        use_mlp = False
        mlp_params = None
        n_hidden = None

    num_rays = rays.directions.shape[0]
    lsp = torch.linspace(0.0, 1.0, num_samples).to(device)
    depths = rays.near[:, None] + lsp[None, :] * (rays.far - rays.near)[:, None]
    tot_num_samples = num_samples + num_samples_inf

    feature_grid = [
        torch.zeros((g), device=device, dtype=torch.float32) for g in output_grid_size
    ]

    weight_grid = [
        torch.zeros((g[:-1]), device=device).unsqueeze(-1) for g in output_grid_size
    ]

    grid_idx = rays.grid_idx
    if num_samples_inf > 0:
        sph = torch.stack(
            [
                _depth_inv_sphere(rays.far, disparity_at_inf, num_samples_inf, step)
                for step in range(num_samples_inf)
            ],
            dim=-1,
        )
        depths = torch.cat([depths, sph], dim=-1)
    samples = depths[..., None] * rays.directions[:, None]
    samples = samples + rays.origins[..., None, :]
    if contract_coords:
        samples = _contract_pi(samples)
    splatting_feature = rays.encoding
    splatting_feature = splatting_feature[:, None, :].expand(-1, tot_num_samples, -1)
    collision_feat = torch.ones_like(splatting_feature[..., 0:1])

    if use_mlp is True:
        weights, bias = _flattened_one_mlp_params_to_list(
            mlp_params.mlp_params,
            n_hidden,
        )
        feature_sampled = sample_grid_list_checkpointed(
            input_grid,
            samples,
            grid_idx,
            mask_out_of_bounds_samples,
            checkpointing=checkpointing,
        )
        feature_sampled = feature_sampled + splatting_feature
        feature_sampled = _eval_mlp(
            feature_sampled,
            weights,
            bias,
            mlp_name="mlp_splatter",
            checkpointing=checkpointing,
        )
    else:
        feature_sampled = splatting_feature

    # NOTE: _splat_shape_rep_checkpoint cannot be checkpointed.
    _splat_shape_rep_checkpoint(
        feature_grid,
        feature_sampled,
        samples,
        grid_idx,
        mask_out_of_bounds_samples,
        False,
    )
    _splat_shape_rep_checkpoint(
        weight_grid,
        collision_feat,
        samples,
        grid_idx,
        mask_out_of_bounds_samples,
        False,
    )
    grid = [
        feature_grid[i] / torch.clamp(weight_grid[i], min=1e-5)
        for i in range(len(feature_grid))
    ]
    if return_list is False:
        grid = torch.cat([g.view(-1, g.shape[-1]) for g in grid], dim=0)
    return grid


def _splat_shape_rep_checkpoint(
    grid: tuple[torch.Tensor, ...],
    splatting_feature: torch.Tensor,  # B x N x C
    samples: torch.Tensor,  # B x N x 3,
    grid_idx: torch.Tensor,  # B
    mask_out_of_bounds_samples: bool,
    checkpointing: bool = False,
):
    """
    Note: This function will throw problems when checkpointing=True
    """
    if checkpointing:
        return checkpoint(
            _splat_shape_rep,
            *(grid, splatting_feature, samples, grid_idx, mask_out_of_bounds_samples),
            use_reentrant=False,
        )
    else:
        return _splat_shape_rep(
            grid, splatting_feature, samples, grid_idx, mask_out_of_bounds_samples
        )


def _splat_shape_rep(
    grid: tuple[torch.Tensor, ...],
    splatting_feature: torch.Tensor,  # B x N x C
    samples: torch.Tensor,  # B x N x 3,
    grid_idx: torch.Tensor,  # B
    mask_out_of_bounds_samples: bool,
):
    used_grids = grid_idx.unique()
    batch_to_idx = [torch.where(grid_idx == i)[0] for i in used_grids]
    samples_list = [samples[idx] for idx in batch_to_idx]
    splatting_feature_list = [splatting_feature[idx] for idx in batch_to_idx]
    samples_padded = torch.nn.utils.rnn.pad_sequence(
        samples_list,
        batch_first=True,
    )
    feature_padded = torch.nn.utils.rnn.pad_sequence(
        splatting_feature_list, batch_first=True
    )
    feature_reshape = feature_padded.view(
        used_grids.shape[0], -1, feature_padded.shape[-1]
    )
    samples_padded_reshape = samples_padded.view(used_grids.shape[0], -1, 3)
    in_bounding_mask = is_in_bounds(samples_padded_reshape).float()
    if mask_out_of_bounds_samples:
        feature_reshape = feature_reshape * in_bounding_mask
    else:
        in_bounding_mask = torch.ones_like(in_bounding_mask)

    in_bounding_mask = in_bounding_mask[..., 0]

    for index, g in enumerate(grid):
        sample_g = g[used_grids]
        assert g.ndim == 5
        n_non_singular_dim = sum(int(s > 1) for s in g.shape[1:-1])

        if n_non_singular_dim == 3:  # 3d voxel grid
            sample_g = _splat_voxel_grid(
                feature_reshape,
                samples_padded_reshape[..., 0],
                samples_padded_reshape[..., 1],
                samples_padded_reshape[..., 2],
                sample_g,
                in_bounding_mask > 0.5
                # mask
            )

            grid[index][used_grids] = sample_g
        elif n_non_singular_dim == 2:  # triplane
            singular_dim = [i for i, s in enumerate(g.shape[1:-1]) if s == 1][0]
            if singular_dim == 0:
                plane = "xy"
            elif singular_dim == 1:
                plane = (
                    "xz"  # the difference with the old lightplane which had zx here !!!
                )
            elif singular_dim == 2:
                plane = "yz"
            else:
                raise ValueError()
            sample_coords = ["xyz".index(c) for c in plane]
            sample_g = _splat_grid(
                feature_reshape,
                samples_padded_reshape[..., sample_coords[0]],
                samples_padded_reshape[..., sample_coords[1]],
                sample_g.squeeze(singular_dim + 1),
                in_bounding_mask > 0.5
                # mask
            )
            grid[index][used_grids] = sample_g.unsqueeze(singular_dim + 1)
        else:
            raise NotImplementedError("no such 3D")


def _splat_3d(
    feature: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    voxel: torch.Tensor,
    W: int,
    H: int,
    D: int,
    C: int,
    B: int,
    mask: torch.Tensor,
):
    x_ = torch.clamp(x, min=0, max=W - 1)
    y_ = torch.clamp(y, min=0, max=H - 1)
    z_ = torch.clamp(z, min=0, max=D - 1)
    feature = feature * (
        (x >= 0) * (x < W) * (y >= 0) * (y < H) * (z < D) * (z >= 0)
    ).unsqueeze(-1)
    batch_size_shift = torch.arange(B, device=voxel.device) * W * H * D
    location = z_ * W * H + y_ * W + x_ + batch_size_shift.unsqueeze(-1)
    location = location[mask].unsqueeze(-1).to(torch.int64)
    feature = feature[mask]

    voxel = voxel.scatter_add(0, location.expand(-1, C), feature)
    return voxel


def _splat_voxel_grid(
    feature: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    voxel: torch.Tensor,
    in_bounding_mask: torch.Tensor,
):
    """
    A function to splat features to planes based on 2D grid splatting.

    Args:
        - feature: [B, N, C], the input feature map.
        - x: [B, N], the x coordinate of the splatting points. It is in the range of [-1, 1].
        - y: [B, N], the y coordinate of the splatting points.  It is in the range of [-1, 1].
        - z: [B, N], the z coordinate of the splatting points.  It is in the range of [-1, 1].
        - plane: [B, D, H, W, C], the output plane.
        - mask: [B, N], in boundary mask
    """
    B, D, H, W, C = voxel.shape
    # convert to image coordinate
    ix = (x + 1.0) / 2.0 * W - 0.5
    iy = (y + 1.0) / 2.0 * H - 0.5
    iz = (z + 1.0) / 2.0 * D - 0.5

    ix0 = ix - ix % 1  # floor
    iy0 = iy - iy % 1  # floor
    iz0 = iz - iz % 1  # floor

    V000x = ix0
    V000y = iy0
    V000z = iz0

    V100x = ix0
    V100y = iy0
    V100z = iz0 + 1

    V010x = ix0
    V010y = iy0 + 1
    V010z = iz0

    V001x = ix0 + 1
    V001y = iy0
    V001z = iz0

    V101x = ix0 + 1
    V101y = iy0
    V101z = iz0 + 1

    V011x = ix0 + 1
    V011y = iy0 + 1
    V011z = iz0

    V110x = ix0
    V110y = iy0 + 1
    V110z = iz0 + 1

    V111x = ix0 + 1
    V111y = iy0 + 1
    V111z = iz0 + 1

    x = ix - ix0
    y = iy - iy0
    z = iz - iz0
    voxel = voxel.view(-1, C)
    voxel = _splat_3d(
        feature * ((1 - x) * (1 - y) * (1 - z)).unsqueeze(-1),
        V000x,
        V000y,
        V000z,
        voxel,
        W,
        H,
        D,
        C,
        B,
        in_bounding_mask,
    )
    voxel = _splat_3d(
        feature * ((1 - x) * (1 - y) * z).unsqueeze(-1),
        V100x,
        V100y,
        V100z,
        voxel,
        W,
        H,
        D,
        C,
        B,
        in_bounding_mask,
    )

    voxel = _splat_3d(
        feature * ((1 - x) * y * (1 - z)).unsqueeze(-1),
        V010x,
        V010y,
        V010z,
        voxel,
        W,
        H,
        D,
        C,
        B,
        in_bounding_mask,
    )

    voxel = _splat_3d(
        feature * ((1 - x) * y * z).unsqueeze(-1),
        V110x,
        V110y,
        V110z,
        voxel,
        W,
        H,
        D,
        C,
        B,
        in_bounding_mask,
    )

    voxel = _splat_3d(
        feature * (x * (1 - y) * (1 - z)).unsqueeze(-1),
        V001x,
        V001y,
        V001z,
        voxel,
        W,
        H,
        D,
        C,
        B,
        in_bounding_mask,
    )

    voxel = _splat_3d(
        feature * (x * (1 - y) * z).unsqueeze(-1),
        V101x,
        V101y,
        V101z,
        voxel,
        W,
        H,
        D,
        C,
        B,
        in_bounding_mask,
    )

    voxel = _splat_3d(
        feature * (x * y * (1 - z)).unsqueeze(-1),
        V011x,
        V011y,
        V011z,
        voxel,
        W,
        H,
        D,
        C,
        B,
        in_bounding_mask,
    )

    voxel = _splat_3d(
        feature * (x * y * z).unsqueeze(-1),
        V111x,
        V111y,
        V111z,
        voxel,
        W,
        H,
        D,
        C,
        B,
        in_bounding_mask,
    )

    return voxel.view(B, D, H, W, C)


def _splat_2d(
    feature: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    plane: torch.Tensor,
    W: int,
    H: int,
    C: int,
    B: int,
    mask: torch.Tensor,
):
    x_ = torch.clamp(x, min=0, max=W - 1)
    y_ = torch.clamp(y, min=0, max=H - 1)
    feature = feature * ((x >= 0) * (x < W) * (y >= 0) * (y < H)).unsqueeze(-1)
    batch_size_shift = torch.arange(B, device=plane.device) * W * H
    location = y_ * W + x_ + batch_size_shift.unsqueeze(-1)
    location = location[mask].unsqueeze(-1).to(torch.int64)
    feature = feature[mask]
    plane = plane.scatter_add(0, location.expand(-1, C), feature)
    return plane


def _splat_grid(
    feature: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    plane: torch.Tensor,
    in_bounding_mask: torch.Tensor,
):
    """
    A function to splat features to planes based on 2D grid splatting.

    Args:
        - feature: [B, C, N], the input feature map.
        - x: [B, N], the x coordinate of the splatting points. It is in the range of [-1, 1].
        - y: [B, N], the y coordinate of the splatting points.  It is in the range of [-1, 1].
        - plane: [1, H, W, C], the output plane. Note it is in shape of [B, H, W, C]
        - mask: [B, N], in boundary mask
    """
    _, H, W, C = plane.shape
    B = x.shape[0]

    # convert to image coordinate
    ix = (x + 1.0) / 2.0 * W - 0.5
    iy = (y + 1.0) / 2.0 * H - 0.5

    ix_nw = ix - ix % 1  # floor
    iy_nw = iy - iy % 1  # floor

    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)
    plane = plane.view(-1, C)
    plane = _splat_2d(
        feature * nw.unsqueeze(-1), ix_nw, iy_nw, plane, W, H, C, B, in_bounding_mask
    )
    plane = _splat_2d(
        feature * ne.unsqueeze(-1), ix_ne, iy_ne, plane, W, H, C, B, in_bounding_mask
    )
    plane = _splat_2d(
        feature * sw.unsqueeze(-1), ix_sw, iy_sw, plane, W, H, C, B, in_bounding_mask
    )
    plane = _splat_2d(
        feature * se.unsqueeze(-1), ix_se, iy_se, plane, W, H, C, B, in_bounding_mask
    )
    return plane.view(B, H, W, C)
