# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy
import math
import time
from typing import List, Optional, Tuple, Union

import torch

from .misc_utils import (
    assert_shape,
    check_grid,
    flatten_grid,
    process_and_flatten_grid,
    unflatten_grid,
)
from .mlp_utils import SplatterParams
from .ray_utils import Rays
from .triton_src import get_lightplane_kernels  # jit-compiles the extension
from .triton_src.shared.const import MIN_BLOCK_SIZE

PROFILE = False
DEBUG = False


def lightplane_splatter(
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
) -> torch.Tensor | list[torch.Tensor]:
    r"""
    This is the functional interface for the Lightplane Splatter.
    For each point along the ray, it splats `rays.encoding` to a zero-initialized
    output grid-list, `output_grid`, where the shape of the output grid-list is
    specified by `output_grid_size`.
    It utilizes the ray-marched splatting, where the `num_samples` 3D points are
    equispaced between `rays.near` and `rays.far`.
    It is useful for directly splatting ray encodings

    It follows::

        ray encoding -> splat -> output_grid


    Args:
        rays: The rays to splat feature.
            It is an instance of `Rays`, including `directions`, `origins`,
            `grid_idx`, `near`, `far`, and `encoding`.
            `grid_idx` indicates the batch index of the 3D grid to sample
            features from::

                x_i = rays.origins[i] + (rays.near[i] + delta_i) * rays.direction[i]

            `endcoding` is the feature to splat for each ray.
        output_grid_size: The sizes of the `output_grid` to be splatted.
            It is a list of tuples, where each tuple is the shape of the
            corresponding grid,in the form `(B, D, H, W, C)`, where `C` has to be
            the same for all output grids.
            Example::

                output_grid_size = [(1, 64, 64, 64, 256), (1, 32, 1, 32, 256)]

        num_samples: The number of points to be splatted along the ray.
            The samples are equispaced between `rays.near` and `rays.far`.
            More specifically, the j-th 3d point `x_ij` along `i-th` ray is
            defined as follows::

                x_ij = rays.origins[i] + (rays.near[i] + j * delta_i) * rays.direction[i],
                    where:
                        delta_i = (rays.far[i] - rays.near[i]) / num_samples

        num_samples_inf: The number of points in the background to be splatted.
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
        return_list: Whether to return a list of grids containing the
            result of the splatting, or a tensor of stacked features.
            Note: Stacked features can be converted to a list of grids with
            the `lightplane.misc_utils.unflatten_grid` function.
        regenerate_code: If `True`, forces the regeneration of the triton code.
        triton_block_size: The block size for Triton. Has to be higher than 16.
        triton_num_warps: The number of warps for Triton.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: The splatted results.

    References:
        [1] MERF: Memory-Efficient Radiance Fields for Real-time View Synthesis in
        Unbounded Scenes, https://arxiv.org/abs/2302.12249

    """
    output_grid_size = torch.tensor(output_grid_size, device=rays.device)
    rays, n_rays_padded = rays.pad_to_block_size(triton_block_size)
    splatting_feature = rays.encoding
    valid_mask = torch.ones_like(splatting_feature[:, 0])
    if n_rays_padded > 0:
        valid_mask[-n_rays_padded:] = 0
    splatted_results = LightplaneSplatterFunction.apply(
        output_grid_size,
        None,
        None,
        None,
        rays.directions,
        rays.origins,
        rays.grid_idx.to(torch.int32),
        rays.near,
        rays.far,
        splatting_feature,
        valid_mask,
        0,
        0,
        # other settings
        num_samples,
        num_samples_inf,
        mask_out_of_bounds_samples,
        contract_coords,
        disparity_at_inf,
        # BS
        triton_block_size,
        triton_num_warps,
        regenerate_code,
    )
    if return_list:
        return unflatten_grid(splatted_results, output_grid_size.to(torch.long))
    else:
        return splatted_results


def lightplane_mlp_splatter(
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
    return_list: bool = True,  # whether return list or stacked tensor
    regenerate_code: bool = False,
    triton_block_size: int = 16,
    triton_num_warps: int = 4,
) -> torch.Tensor | list[torch.Tensor]:
    r"""
    This is the functional interface for the Lightplane Splatter.
    For each point along the ray, it first samples the point feature from the
    corresponding prior input grid `input_grid`, adds the sampled feature to the
    `encoding` of the ray, passes the latter through an MLP, and splats the MLP
    output to the grid-list `output_grid`.
    It utilizes the ray-marched splatting, where the `num_samples` 3D points are
    equispaced between `rays.near` and `rays.far`.

    It follows::

        input_grid -> f_i -> f_i + ray encoding -> mlp -> sf_i ->
        splat -> output_grid

    Args:
        rays: The rays to splat feature.
            It is an instance of `Rays`, including `directions`, `origins`,
            `grid_idx`, `near`, `far`, and `encoding`.
            `grid_idx` indicates the batch index of the 3D grid to sample
            features from::

                x_i = rays.origins[i] + (rays.near[i] + delta_i) * rays.direction[i]

            `endcoding` is the feature to splat for each ray.
        output_grid_size: The sizes of the `output_grid` to be splatted.
            It is a list of tuples, where each tuple is the shape of the
            corresponding grid,in the form `(B, D, H, W, C)`, where `C` has to be
            the same for all output grids.
            Example::

                output_grid_size = [(1, 64, 64, 64, 256), (1, 32, 1, 32, 256)]

        mlp_params: The parameters of the MLP that processes the features
            sampled from `input_grid`.
        input_grid: The grid-list (a list of 3D grids) from which to sample
            features.
            Sames as `grid` in `lightplane_renderer`, it contains `N` grids, where
            each grid is a tensor of shape `(B, D_i, H_i, W_i, C)`.
            `input_grid` should have the same feature dimension (`C`) as
            `rays.encoding` as they are summed together before passing through
            the MLP.

            Similar to `grid` in `lightplane_renderer`, `input_grid` could be
            a 2D tensor (the stacked version of the grid-list).
            In this case, `input_grid_sizes` must be provided to specify the shape
            of `input_grid`.
        num_samples: The number of points to be splatted along the ray.
            The samples are equispaced between `rays.near` and `rays.far`.
            More specifically, the j-th 3d point `x_ij` along `i-th` ray is
            defined as follows::

                x_ij = rays.origins[i] + (rays.near[i] + j * delta_i) * rays.direction[i],
                    where:
                        delta_i = (rays.far[i] - rays.near[i]) / num_samples

        num_samples_inf: The number of points in the background to be splatted.
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
        input_grid_sizes: It specifies the size of `input_grid`.
            It is optional when `input_grid` is a grid-list, but required when
            `input_grid`is a 2D tensor. Example::

                input_grid_sizes = [[B, D_1, H_1, W_1, C], ... , [B, D_N, H_N, W_N, C]]

        return_list: Whether to return a list of grids containing the
            result of the splatting, or a tensor of stacked features.
            Note: Stacked features can be converted to a list of grids with
            the `lightplane.misc_utils.unflatten_grid` function.
        regenerate_code: If `True`, forces the regeneration of the triton code.
        triton_block_size: The block size for Triton. Has to be higher than 16.
        triton_num_warps: The number of warps for Triton.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: The splatted results.

    References:
        [1] MERF: Memory-Efficient Radiance Fields for Real-time View Synthesis in
        Unbounded Scenes, https://arxiv.org/abs/2302.12249

    """
    output_grid_size = torch.tensor(output_grid_size, device=rays.device)
    rays, n_rays_padded = rays.pad_to_block_size(triton_block_size)
    splatting_feature = rays.encoding
    valid_mask = torch.ones_like(splatting_feature[:, 0])
    if n_rays_padded > 0:
        valid_mask[-n_rays_padded:] = 0

    assert (
        len(mlp_params.n_hidden) > 1
    ), f"mlp depth has to be bigger than 1 when using input_grid"
    assert (
        input_grid is not None
    ), f"input_grid cannot be None when mlp_params is not None"
    n_hidden = mlp_params.n_hidden[1].item()
    n_layers = mlp_params.n_hidden.numel() - 1

    check_grid(input_grid, input_grid_sizes)
    input_grid, _, input_grid_size, _ = process_and_flatten_grid(
        input_grid, None, input_grid_sizes, None
    )

    splatted_results = LightplaneSplatterFunction.apply(
        output_grid_size,
        mlp_params.mlp_params,
        input_grid,
        input_grid_size,
        rays.directions,
        rays.origins,
        rays.grid_idx.to(torch.int32),
        rays.near,
        rays.far,
        splatting_feature,
        valid_mask,
        n_hidden,
        n_layers,
        # other settings
        num_samples,
        num_samples_inf,
        mask_out_of_bounds_samples,
        contract_coords,
        disparity_at_inf,
        # BS
        triton_block_size,
        triton_num_warps,
        regenerate_code,
    )
    if return_list:
        return unflatten_grid(splatted_results, output_grid_size.to(torch.long))
    else:
        return splatted_results


class LightplaneSplatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        feature_grid_sizes: torch.Tensor,  # NUM_GRIDS x 5: [[B_1, D_1, H_1, W_1, C_1], ...]
        mlp_params: torch.Tensor | None,  # flattened biases and weights of all mlps
        input_feature_grid: None
        | (
            torch.Tensor
        ),  # NUM_GRIDS * B * D' * H' * W' * C', another gird with same num_grids and batch, for recursive feature updating.
        input_feature_grid_sizes: None
        | (torch.Tensor),  # NUM_GRIDS x 5: [[B_1, D_1, H_1, W_1, C_1], ...]
        directions: torch.Tensor,  # N x 3
        origins: torch.Tensor,  # N x 3
        grid_idx: torch.Tensor,  # N
        near: torch.Tensor,  # N
        far: torch.Tensor,  # N
        splatting_feature: torch.Tensor,  # N x C
        valid_mask: torch.Tensor,  # N
        # mlp sizes
        mlp_dim_hidden: int | None,
        mlp_n_layers: int | None,
        # other settings
        num_samples: int,
        num_samples_inf: int,
        mask_out_of_bounds_samples: bool = False,
        contract_coords: bool = False,
        disparity_at_inf: float = 1e-5,
        # triton block size
        BLOCK_SIZE: int = 16,
        NUM_WARPS: int = 4,
        # force code regeneration
        regenerate_code: bool = False,
    ):
        device = directions.device
        use_mlp = mlp_params is not None

        (
            fw_kernel_feat,
            fw_kernel_weight,
            bw_kernel_feat,
            bw_kernel_weight,
        ) = get_lightplane_kernels(
            "splatter",
            mlp_n_layers,
            regenerate_code=regenerate_code,
        )

        ctx.fw_kernel_feat = fw_kernel_feat
        ctx.fw_kernel_weight = fw_kernel_weight
        ctx.bw_kernel_feat = bw_kernel_feat

        if PROFILE:
            torch.cuda.synchronize()
            time_start = time.time()

        assert BLOCK_SIZE >= MIN_BLOCK_SIZE

        # initialize feature grid
        # All grid should have the same feature dimensions, otherwise cannot be stacked.
        assert all(
            g[-1] == feature_grid_sizes[0, -1] for g in feature_grid_sizes
        ), "All output grids should have the same feature dimensions."
        feature_grid = torch.zeros(
            (
                sum(torch.prod(g[:-1]) for g in feature_grid_sizes),
                feature_grid_sizes[0, -1],
            ),
            device=device,
        ).contiguous()

        # important sizes
        device = feature_grid_sizes.device
        num_grid_channels = feature_grid.shape[-1]
        num_splatting_channels = splatting_feature.shape[-1]
        num_rays = directions.shape[0]
        num_grids = feature_grid_sizes.shape[0]
        grid_batch_size = feature_grid_sizes.shape[1]

        # mlp size params for kernel
        if use_mlp:
            mlp_dim_in = num_splatting_channels
            mlp_dim_out = num_grid_channels

        else:
            mlp_dim_in = num_splatting_channels
            mlp_dim_out = num_splatting_channels
            assert (
                num_grid_channels == num_splatting_channels
            ), f"num_grid_channels should be the same as num_splatting_channels"

        # asserts
        assert_shape(feature_grid_sizes, (num_grids, 5))
        assert_shape(directions, (num_rays, 3))
        assert_shape(origins, (num_rays, 3))
        assert_shape(grid_idx, (num_rays,))
        assert_shape(near, (num_rays,))
        assert_shape(far, (num_rays,))
        assert_shape(splatting_feature, (num_rays, mlp_dim_in))
        assert (
            math.log2(num_grid_channels) % 1 == 0
        ), f"num_grid_channels has to be divided by 2"
        assert (
            num_grid_channels >= MIN_BLOCK_SIZE
        ), f"num_grid_channels has to be bigger than {MIN_BLOCK_SIZE}"
        assert (
            math.log2(num_splatting_channels) % 1 == 0
        ), f"num_splatting_channels has to be divided by 2"
        assert (
            num_splatting_channels >= MIN_BLOCK_SIZE
        ), f"num_splatting_channels has to be bigger than {MIN_BLOCK_SIZE}"
        assert (
            num_rays % BLOCK_SIZE == 0
        ), "We do not support #rays!=multiple of BLOCK_SIZE."

        if use_mlp:
            assert mlp_params is not None
            assert input_feature_grid is not None
            assert input_feature_grid_sizes is not None
            assert mlp_dim_hidden is not None
            assert mlp_n_layers is not None
            assert mlp_params.ndim == 1
            assert splatting_feature.shape[1] == mlp_dim_in
            assert_shape(input_feature_grid_sizes, (num_grids, 5))

        kwargs = {
            # ---- grid ----
            "feature_grid": feature_grid,
            "feature_grid_sizes": feature_grid_sizes,
            # ----- non-differentiable tensors
            "directions": directions,
            "origins": origins,
            "grid_idx": grid_idx,
            "near": near,
            "far": far,
            "splatting_feature": splatting_feature,
            "mask": valid_mask,
            # ----- config keys ----
            "num_samples": num_samples,
            "num_samples_inf": num_samples_inf,
            # ----- sizes ----
            "num_rays": num_rays,
            "grid_channel": num_grid_channels,
            "NUM_GRIDS": num_grids,
            "feature_channel": num_splatting_channels,
            "BLOCK_SIZE": BLOCK_SIZE,
            # ---- switches ----
            "mask_out_of_bounds_samples": int(mask_out_of_bounds_samples),
            "contract_coords": contract_coords,
            "disparity_at_inf": disparity_at_inf,
        }
        if use_mlp:
            kwargs.update(
                {
                    "input_feature_grid": input_feature_grid,
                    "input_feature_grid_sizes": input_feature_grid_sizes,
                    "mlp_params": mlp_params.contiguous(),
                    "DIM_HIDDEN": mlp_dim_hidden,
                    "DIM_IN": mlp_dim_in,
                    "DIM_OUT": mlp_dim_out,
                }
            )
        n_blocks = int(math.ceil(num_rays / BLOCK_SIZE))
        grid = (n_blocks,)
        fw_kernel_feat[grid](num_warps=1 if DEBUG else NUM_WARPS, **kwargs)

        weight_grid = torch.zeros_like(
            feature_grid[..., 0:1]
        )  # single dimension feature grid
        weight_grid_sizes = copy.deepcopy(feature_grid_sizes)
        splating_weight = torch.ones_like(splatting_feature[..., 0:1])
        weight_grid_sizes[:, 4] = 1
        weight_kwargs = {
            # ---- grid ----
            "feature_grid": weight_grid,
            "feature_grid_sizes": weight_grid_sizes,
            # ----- non-differentiable tensors
            "directions": directions,
            "origins": origins,
            "grid_idx": grid_idx,
            "near": near,
            "far": far,
            "splatting_feature": splating_weight,
            "mask": valid_mask,
            # ----- config keys ----
            "num_samples": num_samples,
            "num_samples_inf": num_samples_inf,
            # ----- sizes ----
            "num_rays": num_rays,
            "grid_channel": 1,
            "NUM_GRIDS": num_grids,
            "feature_channel": 1,
            "BLOCK_SIZE": BLOCK_SIZE,
            # ---- switches ----
            "mask_out_of_bounds_samples": int(mask_out_of_bounds_samples),
            "contract_coords": contract_coords,
            "disparity_at_inf": disparity_at_inf,
        }
        fw_kernel_weight[grid](num_warps=1 if DEBUG else NUM_WARPS, **weight_kwargs)

        weight_grid = torch.clamp(weight_grid, min=1e-5)
        # save tensors for bw
        ctx.save_for_backward(
            feature_grid,
            feature_grid_sizes,
            input_feature_grid,
            input_feature_grid_sizes,
            weight_grid,
            mlp_params,
            directions,
            origins,
            grid_idx,
            near,
            far,
            splatting_feature,
            valid_mask,
        )

        # save config keys
        ctx.mlp_dim_in = mlp_dim_in
        ctx.mlp_dim_out = mlp_dim_out
        ctx.mlp_n_layers = mlp_n_layers
        ctx.mlp_dim_hidden = mlp_dim_hidden
        ctx.num_samples = num_samples
        ctx.num_samples_inf = num_samples_inf
        ctx.num_splatting_channels = num_splatting_channels
        ctx.mask_out_of_bounds_samples = mask_out_of_bounds_samples
        ctx.contract_coords = contract_coords
        ctx.disparity_at_inf = disparity_at_inf
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.NUM_WARPS = NUM_WARPS
        ctx.num_grids = num_grids
        ctx.num_rays = num_rays
        ctx.num_grid_channels = num_grid_channels
        ctx.device = device
        ctx.grid_batch_size = grid_batch_size
        ctx.use_mlp = use_mlp

        if PROFILE:
            torch.cuda.synchronize()
            elapsed = time.time() - time_start
            print(f"fw time = {elapsed:1.5f}")

        return feature_grid / weight_grid

    @staticmethod
    def backward(ctx, grad_feature_grid):
        if PROFILE:
            torch.cuda.synchronize()
            time_start = time.time()

        (
            feature_grid,
            feature_grid_sizes,
            input_feature_grid,
            input_feature_grid_sizes,
            weight_grid,
            mlp_params,
            directions,
            origins,
            grid_idx,
            near,
            far,
            splatting_feature,
            valid_mask,
        ) = ctx.saved_tensors

        grad_feature_grid = grad_feature_grid / weight_grid
        device = feature_grid.device

        grad_splatting_feature = torch.zeros_like(splatting_feature)

        n_blocks = int(math.ceil(ctx.num_rays / ctx.BLOCK_SIZE))
        grid = (n_blocks,)

        kwargs = {
            "grad_feature_grid": grad_feature_grid,
            "grad_feature_grid_sizes": feature_grid_sizes,
            # ---- grid ----
            # ----- non-differentiable tensors
            "directions": directions,
            "origins": origins,
            "grid_idx": grid_idx,
            "near": near,
            "far": far,
            "splatting_feature": splatting_feature,
            "mask": valid_mask,
            # ----- config keys ----
            "num_samples": ctx.num_samples,
            "num_samples_inf": ctx.num_samples_inf,
            # ----- sizes ----
            "num_rays": ctx.num_rays,
            "grid_channel": ctx.num_grid_channels,
            "NUM_GRIDS": ctx.num_grids,
            "feature_channel": ctx.num_splatting_channels,
            "BLOCK_SIZE": ctx.BLOCK_SIZE,
            # ---- switches ----
            "mask_out_of_bounds_samples": int(ctx.mask_out_of_bounds_samples),
            "contract_coords": ctx.contract_coords,
            "disparity_at_inf": ctx.disparity_at_inf,
            "grad_splatting_feature": grad_splatting_feature,
        }
        if ctx.use_mlp:
            grad_mlp_params = torch.zeros_like(mlp_params)
            grad_input_feature_grid = torch.zeros_like(input_feature_grid)
            kwargs.update(
                {
                    "feature_grid": feature_grid,
                    "feature_grid_sizes": feature_grid_sizes,
                    "input_feature_grid": input_feature_grid,
                    "input_feature_grid_sizes": input_feature_grid_sizes,
                    "mlp_params": mlp_params.contiguous(),
                    "DIM_HIDDEN": ctx.mlp_dim_hidden,
                    "DIM_IN": ctx.mlp_dim_in,
                    "DIM_OUT": ctx.mlp_dim_out,
                    "grad_mlp_params": grad_mlp_params,
                    "grad_input_feature_grid": grad_input_feature_grid,
                }
            )
        else:
            grad_input_feature_grid = None
            grad_mlp_params = None

        ctx.bw_kernel_feat[grid](num_warps=1 if DEBUG else ctx.NUM_WARPS, **kwargs)

        if PROFILE:
            torch.cuda.synchronize()
            elapsed = time.time() - time_start
            print(f"bw time = {elapsed:1.5f}")

        # TODO: remove for speed
        assert torch.isfinite(grad_splatting_feature).all()

        if grad_mlp_params is not None:
            assert torch.isfinite(grad_mlp_params).all()

        return (
            None,
            grad_mlp_params,
            grad_input_feature_grid,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_splatting_feature,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
