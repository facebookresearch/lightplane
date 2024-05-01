# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import random
from typing import Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from .misc_utils import check_grid_and_color_grid, is_in_bounds, unflatten_grid
from .mlp_utils import DecoderParams, flattened_decoder_params_to_list
from .ray_utils import Rays
from .triton_src.shared.const import MIN_BLOCK_SIZE
from .triton_src.shared.rand_util import int_to_randn_naive

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


def lightplane_renderer_naive(
    rays: Rays,
    grid: tuple[torch.Tensor, ...] | torch.Tensor,
    decoder_params: DecoderParams,
    # ------ config keys ------
    num_samples: int,
    gain: float,
    mask_out_of_bounds_samples: bool = False,
    num_samples_inf: int = 0,
    contract_coords: bool = False,
    inject_noise_sigma: float = 0.0,
    inject_noise_seed: int | None = None,
    disparity_at_inf: float = 1e-5,
    scaffold: torch.Tensor | None = None,
    color_grid: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
    grid_sizes: list[list[int]] | None = None,
    color_grid_sizes: list[list[int]] | None = None,
    triton_num_warps: int = -1,  # ignored, but kept for compatibility with triton api
    triton_block_size: int = -1,  # ignored, but kept for compatibility with triton api
    regenerate_code: bool = False,  # ignored, but kept for compatibility with triton api
    checkpointing: bool = False,  # whether or not use pytorch checkpoint for MLP eval
):
    r"""
    This is the naive implementation of the Lightplane Renderer (`lightplane_renderer`),
    which gives the same numeric results as the Triton implementation with less
    memory efficiency.
    It is useful for debugging and understanding the Triton implementation.
    It outputs the the final color `c`, negative_log_transmittance `T_N` and the expected
    ray-termination length `r` (analogous to depth) of each ray's pixel.

    Its arguments are the same as the Triton implementation in `lightplane_renderer`.
    Additionally, it could work using `torch.torch.utils.checkpoint` by setting
    `checkpointing=True`

    Args:
        rays: The rays to render features.
            It is an instance of `Rays`, including `directions`, `origins`,
            `grid_idx`, `near`, `far`, and `encoding`.
            `grid_idx` indicates the batch index of the 3D grid to sample
            features from::

                x_i = rays.origins[i] + (rays.near[i] + delta_i) * rays.direction[i]

        grid: Grid-list (a list of 3D grids) to sample features from.
            Features are sampled from each 3D grid in the set and summed up as
            the final feature.
            `grid` contains `N` tensors, each with the shape::

                [[B, D_1, H_1, W_1, C], ... , [B, D_N, H_N, W_N, C]]

            Each tensor must have 5 dimensions and all tensors should have
            the same batch size `B` and feature dimension `C`.

            Example:
                If `grid` is a single Voxel grid::

                    grid = [torch.tensor([B, D, H, W, C])]

                If `grid` is a triplane::

                    grid = [
                        torch.tensor([B, 1, H, W, C]),
                        torch.tensor([B, D, 1, W, C]),
                        torch.tensor([B, D, H, 1, C]),
                    ]

            `lightplane_renderer` can also work with `grid` as a 2D tensor,
            which is a stacked tensor from the grid-list `grid`, with the shape
            `[sum_(i=1..N)(B * D_i * H_i * W_i), C]`.
            In this case, the `grid_sizes` must be provided to specify the shape
            of each grid.

            The 2D tensor can be obtained from `lightplane.flatten_grid(grid)`
            to flatten the list of tensors and to also obtain the `grid_sizes`
            argument.
            This is useful for improving memory-effciency when grid-list is
            giant since we internally flatten the grid-list to a 2D tensor.
        decoder_params: The parameters of the decoder, including the MLP
            parameters of `trunk_mlp`, `color_mlp`, and `opacity_mlp`.
        num_samples: The number of sampling points along the ray.
            The samples are equispaced between `rays.near` and `rays.far`.
            More specifically, the j-th 3d point `x_ij` along `i-th` ray is
            defined as follows::

                x_ij = rays.origins[i] + (rays.near[i] + j * delta_i) * rays.direction[i],
                    where:
                        delta_i = (rays.far[i] - rays.near[i]) / num_samples

        gain: A constant to scale the transmittance::

                T_i = exp(-gain * sum_{j=1}^{i} o(x_ij) * delta_i)

        num_samples_inf: The number of background samples along the ray.
            The first background sample is placed at `rays.far`, and the samples
            are spaced in the disparity space until reaching the disparity of
            `disparity_at_inf`.
            More specifically, the j-th background 3d point `b_ij` along `i-th`
            ray is defined as follows::

                b_ij = rays.origins[i] + (rays.far[i] + j * bg_delta_ij) * rays.direction[i],
                    where:
                        bg_delta_ij = 1 / disparity_ij
                        disparity_ij = linspace(1, disparity_at_inf, num_samples_inf)[j]

            These samples are additional to `num_samples`, i.e. the total number
            of samples along a ray is `num_samples + num_samples_inf`.
        mask_out_of_bounds_samples: Whether to mask samples that
            fall outside the [-1, 1] cube (does not apply when contraction with
            `contract_coords` is enabled).
        contract_coords: Whether to map the coordinates of the splatted
            points to always fall into the [-1, 1] cube. The contraction is implemented
            as in MeRF [1]::

                                x[k]                       if |x|_inf <= 1
                contract(x)[k] = x[k] / |x|_inf             if x_k != |x|_inf > 1
                                (2 - 1/x[k]) x_k / |x_k|   if x_k = |x|_inf > 1

            Note: The contraction is useful for representing unbounded scenes.
                E.g. outdoor captures where the scene extends to infinity.
        disparity_at_inf: The disparity value at infinity.
        inject_noise_sigma: The variance of opacity noise to inject.
        inject_noise_seed: The seed of the random noise to inject.
        scaffold: A voxel grid with shape `[B, D, H, W]`, indicating the occupancy
            of the 3D space. If provided, the renderer will only render the points
            that are not empty in the scaffold.
        color_grid: Another grid-list (a list of 3D grids) storing color features.
            If provided, the renderer will regress the color from features
            sampled from `color_grid`, using `color_mlp`.

            Similar to `grid`, `color_grid` could also be a 2D tensor with
            `color_grid_sizes` provided.
            `color_grid` should be the same type as `grid`.
        grid_sizes: It specifies the size of `grid`.
            It is optional when `grid` is a grid-list, but required when `grid` is a 
            2D tensor. Example::

                grid_sizes = [[B, D_1, H_1, W_1, C], ... , [B, D_N, H_N, W_N, C]]

        color_grid_sizes: It specifies the size of `color_grid` when `color_grid`
             is a 2D tensor.
            It is optional when `color_grid` is a grid-list, but required 
            when `color_grid` is a 2D tensor. Example::

                color_grid_sizes = [[B, D_1, H_1, W_1, C], ... , [B, D_N, H_N, W_N, C]]

        regenerate_code: Ignored, but kept for compatibility with triton api.
        triton_block_size: Ignored, but kept for compatibility with triton api.
        triton_num_warps: Ignored, but kept for compatibility with triton api.
        checkpointing: Whether or not use `torch.utils.checkpoint` for checkpointing.
    Returns:
        ray_length_render: The rendered ray-termination length `r` (i.e. distance along the ray).
        negative_log_transmittances: The negative log transmittances of the ray.
        feature_render: The expected features of the ray.

    References:
        [1] MERF: Memory-Efficient Radiance Fields for Real-time View Synthesis in
        Unbounded Scenes, https://arxiv.org/abs/2302.12249
    """
    grid, color_grid, grid_sizes, color_grid_sizes = check_grid_and_color_grid(
        grid, color_grid, grid_sizes, color_grid_sizes
    )

    # if grid is flatten tensor, we need to unflatten them to 5-dim tensor so
    # that pytorch can interpolate on them.
    # unflatten use split operations, which should use no addtional memories as
    # it creats tensor view instead of allocating memories.
    if isinstance(grid, torch.Tensor):
        grid_sizes_tensor = torch.tensor(
            grid_sizes, device=grid.device, dtype=torch.long
        )
        grid = unflatten_grid(grid, grid_sizes_tensor)
        if color_grid is not None:
            color_grid_sizes_tensor = torch.tensor(
                color_grid_sizes, device=color_grid.device, dtype=torch.long
            )
            color_grid = unflatten_grid(color_grid, color_grid_sizes_tensor)

    device = rays.device
    num_rays = rays.directions.shape[0]
    lsp = torch.linspace(0.0, 1.0, num_samples).to(device)
    depths = rays.near[:, None] + lsp[None, :] * (rays.far - rays.near)[:, None]
    tot_num_samples = num_samples + num_samples_inf

    if inject_noise_seed is None:
        if inject_noise_sigma > 0.0:
            inject_noise_seed = int(random.randint(0, 1000000))
        else:
            inject_noise_seed = 0

    if inject_noise_sigma > 0.0:
        inject_opacity_noise = _get_sample_randn(
            tot_num_samples,
            num_rays,
            device,
            inject_noise_seed,
        )
        inject_opacity_noise = inject_opacity_noise * inject_noise_sigma
    else:
        inject_opacity_noise = None

    if num_samples_inf > 0:
        sph = torch.stack(
            [
                _depth_inv_sphere(rays.far, disparity_at_inf, num_samples_inf, step)
                for step in range(num_samples_inf)
            ],
            dim=-1,
        )
        depths = torch.cat([depths, sph], dim=-1)

    points = depths[..., None] * rays.directions[:, None]
    points = points + rays.origins[..., None, :]

    delta_one = (
        (rays.far - rays.near) / (num_samples - 1)
        if num_samples > 1
        else torch.ones_like(rays.near)
    )
    delta = torch.cat([delta_one[:, None], depths.diff(dim=-1)], dim=-1)

    if VERBOSE:
        print("near")
        print(rays.near)
        print("far")
        print(rays.far)
        print("depths")
        print(depths)
        print("delta")
        print(delta)
        print("centers")
        print(rays.origins)

    # if checkpointing:
    #     opacity, color = checkpoint(
    #         lightplane_eval_mlp,
    #         *(
    #             samples,
    #             grid,
    #             rays.grid_idx,
    #             decoder_params,
    #             rays.encoding,
    #             gain,
    #             mask_out_of_bounds_samples,
    #             inject_opacity_noise,
    #             scaffold,
    #             color_grid
    #         ),
    #         use_reentrant=False
    #     )
    # else:
    opacity, color = lightplane_eval_mlp(
        points,
        grid,
        rays.grid_idx,
        decoder_params,
        rays.encoding,  # ..., C
        gain,
        mask_out_of_bounds_samples=mask_out_of_bounds_samples,
        inject_opacity_noise=inject_opacity_noise,
        scaffold=scaffold,
        color_grid=color_grid,
        checkpointing=checkpointing,
        contract_coords=contract_coords,
    )
    delta_opacity = opacity * delta
    delta_opacity = torch.nn.functional.pad(delta_opacity, (1, 0))

    negative_log_transmittances = torch.cumsum(delta_opacity, dim=-1)
    transmittance = torch.exp(-negative_log_transmittances)

    rweights = -transmittance.diff(dim=-1)
    if VERBOSE:
        print("weight")
        print(rweights)

    ray_length_render = (depths * rweights).sum(dim=-1)
    feature_render = (color * rweights[..., None]).sum(-2)
    negative_log_transmittance = negative_log_transmittances[..., -1]

    if decoder_params.color_chn < feature_render.shape[-1]:
        feature_render = feature_render[..., : decoder_params.color_chn]

    return (
        ray_length_render,
        negative_log_transmittance,
        feature_render,
    )


def lightplane_eval_mlp(
    points: torch.Tensor,  # R x N x 3; packed with ray_grid_idx
    grid: tuple[torch.Tensor, ...],
    ray_grid_idx: torch.Tensor,
    decoder_params: DecoderParams,
    rays_encoding: torch.Tensor,
    gain: float,
    mask_out_of_bounds_samples: bool = False,
    inject_opacity_noise: torch.Tensor | None = None,
    scaffold: torch.Tensor | None = None,
    color_grid: tuple[torch.Tensor, ...] | None = None,
    checkpointing: bool = False,  # whether or not use pytorch checkpoint for MLP eval
    contract_coords: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert points.ndim >= 3
    (
        weights_trunk,
        biases_trunk,
        weights_opacity,
        biases_opacity,
        weights_color,
        biases_color,
    ) = flattened_decoder_params_to_list(
        decoder_params.mlp_params,
        decoder_params.n_hidden_trunk,
        decoder_params.n_hidden_opacity,
        decoder_params.n_hidden_color,
    )

    if VERBOSE:
        print("w_trunk")
        for w in weights_trunk:
            print(w)
        print("b_trunk")
        for b in biases_trunk:
            print(b)

    if contract_coords:
        points = _contract_pi(points)
    feature_sampled = sample_grid_list_checkpointed(
        grid,
        points,
        ray_grid_idx,
        mask_out_of_bounds_samples,
        checkpointing=checkpointing,
    )

    if color_grid is not None:
        feature_sampled_color = sample_grid_list_checkpointed(
            color_grid,
            points,
            ray_grid_idx,
            mask_out_of_bounds_samples,
            checkpointing=checkpointing,
        )
    else:
        feature_sampled_color = None

    if VERBOSE:
        print("feature_sampled")
        print(feature_sampled)

    if feature_sampled_color is None:
        # we have a single feature grid
        feature_trunk = _eval_mlp(
            feature_sampled,
            weights_trunk,
            biases_trunk,
            mlp_name="trunk",
            checkpointing=checkpointing,
        )
        feature_trunk = torch.relu(feature_trunk)

        if VERBOSE:
            print("feature_trunk")
            print(feature_trunk)

        opacity_raw = _eval_mlp(
            feature_trunk,
            weights_opacity,
            biases_opacity,
            mlp_name="opacity",
            checkpointing=checkpointing,
        )

        if VERBOSE:
            print("opacity_raw")
            print(opacity_raw)

        ray_feature_trunk = feature_trunk + rays_encoding[:, None]

        if VERBOSE:
            print("ray_feature_trunk")
            print(ray_feature_trunk)
        log_color = _eval_mlp(
            ray_feature_trunk,
            weights_color,
            biases_color,
            mlp_name="color",
            checkpointing=checkpointing,
        )

        if VERBOSE:
            print("log_color")
            print(log_color)

    else:
        # we use a relu right after sampling (i.e. a relu-field)
        feature_sampled = torch.relu(feature_sampled)
        feature_sampled_color = torch.relu(feature_sampled_color)
        ray_feature_sampled_color = feature_sampled_color + rays_encoding[:, None]

        if VERBOSE:
            print("feature_sampled")
            print(feature_sampled)
            print("feature_sampled_color")
            print(feature_sampled_color)

        assert len(weights_trunk) == 0
        assert len(biases_trunk) == 0
        opacity_raw = _eval_mlp(
            feature_sampled,
            weights_opacity,
            biases_opacity,
            mlp_name="opacity",
            checkpointing=checkpointing,
        )

        if VERBOSE:
            print("opacity_raw")
            print(opacity_raw)

        log_color = _eval_mlp(
            ray_feature_sampled_color,
            weights_color,
            biases_color,
            mlp_name="color",
            checkpointing=checkpointing,
        )

        if VERBOSE:
            print("log_color")
            print(log_color)

    assert opacity_raw.shape[-1] == 1
    opacity_raw = opacity_raw[..., 0]
    if inject_opacity_noise is not None:
        if VERBOSE:
            print("inject_opacity_noise")
            print(inject_opacity_noise)
        opacity_raw = opacity_raw + inject_opacity_noise
    opacity = gain * torch.nn.functional.softplus(opacity_raw)

    # TODO: allow for output without activation
    feature_out = torch.sigmoid(log_color)

    if scaffold is not None:
        scaffold_value = sample_grid_list_checkpointed(
            (scaffold[..., None],),
            points,
            ray_grid_idx,
            True,
            mode="nearest",
            checkpointing=checkpointing,
        )
        if VERBOSE:
            print("scaffold_value")
            print(scaffold_value)
        prev_opacity = opacity

        opacity = opacity * scaffold_value[..., 0]
        feature_out = feature_out * scaffold_value

    return opacity, feature_out


def lightplane_eval_mlp_opacity_only(
    points: torch.Tensor,  # R x N x 3; packed with ray_grid_idx
    grid: tuple[torch.Tensor, ...],
    ray_grid_idx: torch.Tensor,
    decoder_params: DecoderParams,
    gain: float,
    mask_out_of_bounds_samples: bool = False,
    inject_opacity_noise: torch.Tensor | None = None,
    scaffold: torch.Tensor | None = None,
    checkpointing: bool = False,  # whether or not use pytorch checkpoint for MLP eval
    contract_coords: bool = False,
) -> torch.Tensor:
    assert points.ndim >= 3
    (
        weights_trunk,
        biases_trunk,
        weights_opacity,
        biases_opacity,
        weights_color,
        biases_color,
    ) = flattened_decoder_params_to_list(
        decoder_params.mlp_params,
        decoder_params.n_hidden_trunk,
        decoder_params.n_hidden_opacity,
        decoder_params.n_hidden_color,
    )

    if VERBOSE:
        print("w_trunk")
        for w in weights_trunk:
            print(w)
        print("b_trunk")
        for b in biases_trunk:
            print(b)

    feature_sampled = sample_grid_list_checkpointed(
        grid,
        points,
        ray_grid_idx,
        mask_out_of_bounds_samples,
        checkpointing=checkpointing,
    )

    if VERBOSE:
        print("feature_sampled")
        print(feature_sampled)

    # we have a single feature grid
    feature_trunk = _eval_mlp(
        feature_sampled,
        weights_trunk,
        biases_trunk,
        mlp_name="trunk",
        checkpointing=checkpointing,
    )
    feature_trunk = torch.relu(feature_trunk)

    if VERBOSE:
        print("feature_trunk")
        print(feature_trunk)

    opacity_raw = _eval_mlp(
        feature_trunk,
        weights_opacity,
        biases_opacity,
        mlp_name="opacity",
        checkpointing=checkpointing,
    )

    if VERBOSE:
        print("opacity_raw")
        print(opacity_raw)

    assert opacity_raw.shape[-1] == 1
    opacity_raw = opacity_raw[..., 0]
    if inject_opacity_noise is not None:
        if VERBOSE:
            print("inject_opacity_noise")
            print(inject_opacity_noise)
        opacity_raw = opacity_raw + inject_opacity_noise
    opacity = gain * torch.nn.functional.softplus(opacity_raw)
    if scaffold is not None:
        scaffold_value = sample_grid_list_checkpointed(
            (scaffold[..., None],),
            points,
            ray_grid_idx,
            True,
            mode="nearest",
            checkpointing=checkpointing,
        )
        if VERBOSE:
            print("scaffold_value")
            print(scaffold_value)
        opacity = opacity * scaffold_value[..., 0]
        feature_out = feature_out * scaffold_value

    return opacity


def sample_grid_list_checkpointed(
    grid: tuple[torch.Tensor, ...],
    points: torch.Tensor,  # B x N x 3
    grid_idx: torch.Tensor,  # B
    mask_out_of_bounds_samples: bool,
    mode="bilinear",
    checkpointing=False,
) -> torch.Tensor:  # B x N x C
    if checkpointing:
        return checkpoint(
            _sample_grid_list,
            *(grid, points, grid_idx, mask_out_of_bounds_samples, mode),
            use_reentrant=False,
        )
    else:
        return _sample_grid_list(
            grid, points, grid_idx, mask_out_of_bounds_samples, mode
        )


def _sample_grid_list(
    grid: tuple[torch.Tensor, ...],
    points: torch.Tensor,  # B x N x 3
    grid_idx: torch.Tensor,  # B
    mask_out_of_bounds_points: bool,
    mode="bilinear",
) -> torch.Tensor:  # B x N x C
    used_grids = grid_idx.unique()
    batch_to_idx = [torch.where(grid_idx == i)[0] for i in used_grids]
    points_list = [points[idx] for idx in batch_to_idx]
    points_padded = torch.nn.utils.rnn.pad_sequence(
        points_list,
        batch_first=True,
    )

    sampled_padded = sum(
        _sample_one_grid(
            g[used_grids],
            points_padded,
            mask_out_of_bounds_points,
            mode=mode,
        )
        for g in grid
    )

    assert sampled_padded.shape[:-1] == points_padded.shape[:-1]

    sampled_list = torch.nn.utils.rnn.unpad_sequence(
        sampled_padded,
        torch.tensor([len(l) for l in batch_to_idx]),
        batch_first=True,
    )

    if VERBOSE:
        print("points")
        print(points)

    sampled = torch.zeros(
        points.shape[0],
        points.shape[1],
        sampled_list[0].shape[-1],
        device=points.device,
        dtype=points.dtype,
    )
    sampled[torch.cat(batch_to_idx)] = torch.cat(sampled_list, dim=0)

    return sampled


def _sample_one_grid(
    g: torch.Tensor, points: torch.Tensor, mask_out_of_bounds_samples: bool, mode: str
):
    assert g.ndim == 5, "We support only B x D x H x W x C grids for now."
    n_non_singular_dim = sum(int(s > 1) for s in g.shape[1:-1])
    if n_non_singular_dim == 3:  # 3d voxel grid
        sampled = torch.nn.functional.grid_sample(
            g.permute(0, 4, 1, 2, 3),
            points[..., None, :],
            align_corners=False,
            mode=mode,
            # mode="nearest",
        )[..., 0].permute(0, 2, 3, 1)
    elif n_non_singular_dim == 2:  # triplane
        singular_dim = [i for i, s in enumerate(g.shape[1:-1]) if s == 1][0]
        if singular_dim == 0:
            plane = "xy"
        elif singular_dim == 1:
            plane = "xz"
        elif singular_dim == 2:
            plane = "yz"
        else:
            raise ValueError()
        sample_coords = ["xyz".index(c) for c in plane]

        sampled = torch.nn.functional.grid_sample(
            g.squeeze(singular_dim + 1).permute(0, 3, 1, 2),
            points[..., sample_coords],
            align_corners=False,
            mode=mode,
        ).permute(0, 2, 3, 1)

        # if True:  # debug, TODO: move to separate test
        #     _, ID, IH, IW, _ = g.shape
        #     mask_ = (torch.tensor([IW, IH, ID], device=points.device) > 1).float()
        #     points_ = points * mask_
        #     sampled_ = torch.nn.functional.grid_sample(
        #         g.permute(0, 4, 1, 2, 3),
        #         points_[..., None, :],
        #         align_corners=False,
        #         mode=mode,
        #     )[..., 0].permute(0, 2, 3, 1)
        #     assert torch.allclose(sampled, sampled_, atol=1e-5, rtol=1e-5)

        if VERBOSE:
            print(f"sampled {plane}[0]:")
            print(sampled[0])

    else:
        raise ValueError(
            f"Unexpected n non-singulare dim of input grid ({n_non_singular_dim})"
        )

    if mask_out_of_bounds_samples:
        in_bounds_mask = is_in_bounds(points)
        sampled = sampled * in_bounds_mask.float()

    return sampled


def _eval_mlp(
    vec: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    biases: tuple[torch.Tensor, ...],
    mlp_name: str = "",
    checkpointing: bool = False,
):
    if checkpointing:
        return _eval_mlp_checkpointing(vec, weights, biases, mlp_name)
    else:
        return _eval_mlp_org(vec, weights, biases, mlp_name)


def _eval_mlp_checkpointing(
    vec: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    biases: tuple[torch.Tensor, ...],
    mlp_name: str = "",
):
    return checkpoint(
        _eval_mlp_org, *(vec, weights, biases, mlp_name), use_reentrant=False
    )


def _eval_mlp_org(
    vec: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    biases: tuple[torch.Tensor, ...],
    mlp_name: str = "",
):
    n_l = len(weights)
    assert n_l == len(biases)
    for l in range(n_l):
        vec = vec @ weights[l] + biases[l]
        if VERBOSE:
            if mlp_name == "trunk" and l == 0:
                print(weights[l])
                print(biases[l])
            print(f"x{l}@w+b")
            print(vec[0])
        if l < n_l - 1:
            vec = torch.relu(vec)
    return vec


def _get_sample_randn(
    num_samples,
    num_rays,
    device,
    inject_noise_seed,
):
    num_rays_pad = max(num_rays, MIN_BLOCK_SIZE)
    i1 = (
        num_samples * torch.arange(num_rays, device=device)[:, None]
        + torch.arange(num_samples, device=device)[None]
        + 1
    ).long()
    i2 = i1 + num_rays_pad * num_samples
    r = int_to_randn_naive(i1.reshape(-1), i2.reshape(-1), inject_noise_seed)
    return r.reshape(num_rays, num_samples)


def _contract_pi(x):
    n = x.abs().max(dim=-1).values[..., None]
    x_contract = torch.where(
        n <= 1.0,
        x,
        torch.where(
            (x.abs() - n).abs() <= 1e-7,
            (2 - 1 / x.abs()) * (x / x.abs()),
            x / n,
        ),
    )
    return x_contract / 2


def _depth_inv_sphere(far, disparity_at_inf, n, step):
    frac_step = (step + 1) / n
    n_disp = (disparity_at_inf - 1) * frac_step + 1
    return far * (1 / n_disp)
