# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import math
import random
import time
import warnings
from typing import List, Optional, Tuple

import torch

from .misc_utils import (
    assert_shape,
    check_grid_and_color_grid,
    flatten_grid,
    process_and_flatten_grid,
)
from .mlp_utils import DecoderParams, get_triton_function_input_dims
from .ray_utils import Rays
from .triton_src import get_lightplane_kernels
from .triton_src.shared.const import MIN_BLOCK_SIZE

PROFILE = False
DEBUG = False


def lightplane_renderer(
    rays: Rays,
    grid: tuple[torch.Tensor, ...] | torch.Tensor,
    decoder_params: DecoderParams,
    # ------ config keys ------
    num_samples: int,
    gain: float,
    num_samples_inf: int = 0,
    mask_out_of_bounds_samples: bool = False,
    contract_coords: bool = False,
    disparity_at_inf: float = 1e-5,
    inject_noise_sigma: float = 0.0,
    inject_noise_seed: int | None = None,
    scaffold: torch.Tensor | None = None,
    color_grid: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
    grid_sizes: list[list[int]] | None = None,
    color_grid_sizes: list[list[int]] | None = None,
    regenerate_code: bool = False,
    triton_block_size: int = 16,
    triton_num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    This is the main functional interface for the Lightplane Renderer.
    It outputs the the final color `c`, negative_log_transmittance `T_N` and the expected
    ray-termination length `r` (analogous to depth) of each ray's pixel.

    For `N=num_samples` equispaced 3D points `pt_3d_i` between the `near` and
    `far` ray-lengths, it samples the feature `f_i(x_i)` of 3D point
    `pt_3d_i =x_i` from the grid-list `grid` and calculate its corresponding
    renderering results.

    There are three MLPs: `trunk_mlp`, `color_mlp` and `opacity_mlp`, whose
    parameters are specified in `decoder_params`.

    - trunk_mlp: Regresses the features from the `f_i(x_i)`::

        e_i(x_i) = trunk_mlp(f_i(x_i))

    - color_mlp: Regresses target features (e.g. final colors) from the
      output of `trunk_mlp` with ray encoding as additional input.
      The dimension of the output is `color_chn`::

        c_i(x_i) = color_mlp(e_i(x_i) + ray_encoding)

    - opacity_mlp: Regresses the opacity scalar from the output of `trunk_mlp`::

        o_i(x_i) = opacity_mlp(e_i(x_i))

    Args:
        rays: The rays to render features.
            It is an instance of `Rays`, with fields `directions`, `origins`,
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

            Note:
                The 2D tensor can be obtained from `lightplane.flatten_grid(grid)`
                to flatten the list of tensors and to also obtain the `grid_sizes`
                argument.

            Note:
                Using 2D tensor inputs improves memory-effciency when grid-list is
                large in memory.
        decoder_params: The parameters of the decoder MLPs:
            `trunk_mlp`, `color_mlp`, and `opacity_mlp`.
        num_samples: The number of sampled points along the ray.
            The samples are equispaced between `rays.near` and `rays.far`.
            More specifically, the `j`-th 3d point `x_ij` along `i-th` ray is
            defined as follows::

                x_ij = rays.origins[i] + (rays.near[i] + j * delta_i) * rays.direction[i],
                    where:
                        delta_i = (rays.far[i] - rays.near[i]) / num_samples

        gain: A constant to scale the transmittance `T_i` of `i`-the point along a ray::

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
        contract_coords: Whether to map the coordinates of the rendered
            points to always fall into the [-1, 1] cube. The contraction is implemented
            as in MeRF [1]::

                                x[k]                       if |x|_inf <= 1
                contract(x)[k] = x[k] / |x|_inf             if x_k != |x|_inf > 1
                                (2 - 1/x[k]) x_k / |x_k|   if x_k = |x|_inf > 1

            Note:
                The contraction is useful for representing unbounded scenes.
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
            It is optional when `grid` is a grid-list, but required when `grid`
            is a 2D tensor. Example::

                grid_sizes = [[B, D_1, H_1, W_1, C], ... , [B, D_N, H_N, W_N, C]].

        color_grid_sizes: It specifies the size of `color_grid` when `color_grid`
            is a 2D tensor.
            It is optional when `color_grid` is a grid-list, but required when
            `color_grid`is a 2D tensor. Example::

                color_grid_sizes = [[B, D_1, H_1, W_1, C], ... , [B, D_N, H_N, W_N, C]]

        regenerate_code: If `True`, forces the regeneration of the triton code.
        triton_block_size: The block size for Triton. Has to be higher than 16.
        triton_num_warps: The number of warps for Triton.

    Returns:
        ray_length_render: The rendered ray-termination length `r (i.e. distance along the ray).

        negative_log_transmittances: The negative log transmittances of the ray.

        feature_render: The rendered features of the ray.

    References:
        [1] MERF: Memory-Efficient Radiance Fields for Real-time View Synthesis in
        Unbounded Scenes, https://arxiv.org/abs/2302.12249
    """
    grid, color_grid, grid_sizes, color_grid_sizes = check_grid_and_color_grid(
        grid, color_grid, grid_sizes, color_grid_sizes
    )
    grid, color_grid, grid_sizes, color_grid_sizes = process_and_flatten_grid(
        grid, color_grid, grid_sizes, color_grid_sizes
    )

    rays, n_rays_padded = rays.pad_to_block_size(triton_block_size)

    (
        mlp_dim_hidden_trunk,
        mlp_dim_hidden_opacity,
        mlp_dim_hidden_color,
        mlp_n_layers_trunk,
        mlp_n_layers_opacity,
        mlp_n_layers_color,
        color_chn_triton,
    ) = get_triton_function_input_dims(
        decoder_params.n_hidden_trunk,
        decoder_params.n_hidden_opacity,
        decoder_params.n_hidden_color,
    )

    color_chn = decoder_params.color_chn

    if inject_noise_sigma > 0.0:
        if inject_noise_seed is None:
            inject_noise_seed = int(random.randint(0, 1000000))
    else:
        inject_noise_seed = 0

    (
        ray_length_render,
        negative_log_transmittances,
        feature_render,
    ) = LightplaneFunction.apply(
        grid,
        grid_sizes,
        decoder_params.mlp_params,
        rays.directions,
        rays.origins,
        rays.grid_idx.to(torch.int32),
        rays.near,
        rays.far,
        rays.encoding,
        scaffold,
        color_grid,
        color_grid_sizes,
        # mlp sizes
        mlp_dim_hidden_trunk,
        mlp_dim_hidden_opacity,
        mlp_dim_hidden_color,
        mlp_n_layers_trunk,
        mlp_n_layers_opacity,
        mlp_n_layers_color,
        # other settings
        num_samples,
        num_samples_inf,
        gain,
        color_chn_triton,
        mask_out_of_bounds_samples,
        contract_coords,
        disparity_at_inf,
        inject_noise_sigma,
        inject_noise_seed,
        # BS
        triton_block_size,
        triton_num_warps,
        regenerate_code,
    )

    # crop the features to the requested number of channels
    if color_chn_triton > color_chn:
        feature_render = feature_render[:, :color_chn]

    if n_rays_padded > 0:
        ray_length_render, negative_log_transmittances, feature_render = (
            t[:-n_rays_padded]
            for t in [ray_length_render, negative_log_transmittances, feature_render]
        )

    return ray_length_render, negative_log_transmittances, feature_render


class LightplaneFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        feature_grid: torch.Tensor,  # NUM_GRIDS * B * D * H * W * C
        feature_grid_sizes: torch.Tensor,  # NUM_GRIDS x 5: [[B_1, D_1, H_1, W_1, C_1], ...]
        mlp_params: torch.Tensor,  # flattened biases and weights of all mlps
        directions: torch.Tensor,  # N x 3
        origins: torch.Tensor,  # N x 3
        grid_idx: torch.Tensor,  # N
        near: torch.Tensor,  # N
        far: torch.Tensor,  # N
        ray_encoding: torch.Tensor,  # N x C
        scaffold: torch.Tensor | None,  # B x D x H x W x 1
        color_feature_grid: torch.Tensor | None,  # NUM_GRIDS * B * D * H * W * C
        color_feature_grid_sizes: None
        | (torch.Tensor),  # NUM_GRIDS x 5: [[B_1, D_1, H_1, W_1, C_1], ...]
        # mlp sizes
        mlp_dim_hidden_trunk: int,
        mlp_dim_hidden_opacity: int,
        mlp_dim_hidden_color: int,
        mlp_n_layers_trunk: int,
        mlp_n_layers_opacity: int,
        mlp_n_layers_color: int,
        # other settings
        num_samples: int,
        num_samples_inf: int,
        gain: float,
        num_render_channels: int,
        mask_out_of_bounds_samples: bool = False,
        contract_coords: bool = False,
        disparity_at_inf: float = 1e-5,
        inject_noise_sigma: float = 0.0,
        inject_noise_seed: int = 0,
        # triton block size
        BLOCK_SIZE: int = 16,
        NUM_WARPS: int = 4,
        # force code regeneration
        regenerate_code: bool = False,
    ):
        fw_kernel, bw_kernel = get_lightplane_kernels(
            "renderer",
            mlp_n_layers_trunk,
            mlp_n_layers_opacity,
            mlp_n_layers_color,
            regenerate_code=regenerate_code,
        )

        ctx.fw_kernel = fw_kernel
        ctx.bw_kernel = bw_kernel

        if PROFILE:
            torch.cuda.synchronize()
            time_start = time.time()

        use_separate_color_grid = color_feature_grid is not None

        assert (
            BLOCK_SIZE >= MIN_BLOCK_SIZE
        ), f"BLOCK_SIZE has to be bigger than {MIN_BLOCK_SIZE}"
        assert (
            num_render_channels >= MIN_BLOCK_SIZE
        ), f"num_render_channels has to be bigger than {MIN_BLOCK_SIZE}"

        if mask_out_of_bounds_samples and contract_coords:
            warnings.warn(
                "The renderer has been configured to contract the coordinates"
                " lying outside the [-1,1] cube (contract_coords=True)"
                " and to also mask out all such points"
                " (mask_out_of_bounds_samples=True)."
            )

        # important sizes
        device = feature_grid.device
        num_grid_channels = feature_grid.shape[-1]
        num_rays = directions.shape[0]
        num_grids = feature_grid_sizes.shape[0]
        grid_batch_size = feature_grid_sizes[0, 0].item()
        assert (feature_grid_sizes[:, 0] == grid_batch_size).all()
        assert all(gs[-1] == num_grid_channels for gs in feature_grid_sizes)
        grid_spatial_numel = feature_grid_sizes[:, :-1].prod(dim=1).sum()
        assert grid_spatial_numel == feature_grid.numel() // num_grid_channels
        feature_grid_sizes.shape[:-1].numel()

        # https://github.com/openai/triton/issues/2688#issue-2003537756

        # mlp size params for kernel:
        if use_separate_color_grid:
            assert (
                mlp_n_layers_trunk == 0
            ), f"mlp_n_layers_trunk has to be 0 when use_separate_color_grid"
            assert (
                mlp_dim_hidden_trunk == 0
            ), f"mlp_dim_hidden_trunk has to be 0 when use_separate_color_grid"
            dim_out_trunk = 0
            dim_in_trunk = 0
            dim_in_color = num_grid_channels
            dim_in_opacity = num_grid_channels
            dim_out_color = num_render_channels

        else:
            dim_out_trunk = mlp_dim_hidden_trunk
            dim_in_trunk = num_grid_channels
            dim_in_color = dim_out_trunk
            dim_in_opacity = dim_out_trunk
            dim_out_color = num_render_channels

        # asserts
        assert_shape(feature_grid_sizes, (num_grids, 5))
        assert_shape(directions, (num_rays, 3))
        assert_shape(origins, (num_rays, 3))
        assert_shape(grid_idx, (num_rays,))
        assert_shape(near, (num_rays,))
        assert_shape(far, (num_rays,))
        assert_shape(ray_encoding, (num_rays, dim_in_color))
        assert (
            math.log2(num_grid_channels) % 1 == 0
        ), f"num_grid_channels has to be a power of 2"
        assert (
            num_grid_channels >= MIN_BLOCK_SIZE
        ), f"num_grid_channels has to be bigger than {MIN_BLOCK_SIZE}"
        assert (
            math.log2(num_render_channels) % 1 == 0
        ), f"num_render_channels has to be a power of 2"
        assert (
            num_render_channels >= MIN_BLOCK_SIZE
        ), f"num_render_channels has to be bigger than {MIN_BLOCK_SIZE}"
        assert mlp_params.ndim == 1
        assert (
            ray_encoding.shape[1] == dim_in_color
        ), f"ray_encoding should have the same dimension as dim_in_color"
        assert (
            num_rays % BLOCK_SIZE == 0
        ), "We do not support num_rays!=multiple of BLOCK_SIZE."

        if use_separate_color_grid:
            num_color_grids = color_feature_grid_sizes.shape[0]
            assert_shape(color_feature_grid_sizes, (num_color_grids, 5))
            assert color_feature_grid.shape[-1] == num_grid_channels
        else:
            assert (
                color_feature_grid_sizes is None
            ), "color_feature_grid_sizes has to be None when use_separate_color_grid is False"
            color_feature_grid_sizes = torch.empty(
                (1,), dtype=torch.int32, device=device
            )
            color_feature_grid = torch.empty((1,), dtype=torch.float32, device=device)
            num_color_grids = 0

        # check the number of mlp param elems is correct
        numel_params_trunk = _get_mlp_n_params(
            dim_in_trunk, mlp_dim_hidden_trunk, dim_out_trunk, mlp_n_layers_trunk
        )
        numel_params_opacity = _get_mlp_n_params(
            dim_in_opacity, mlp_dim_hidden_opacity, 1, mlp_n_layers_opacity
        )
        numel_params_color = _get_mlp_n_params(
            dim_in_color, mlp_dim_hidden_color, dim_out_color, mlp_n_layers_color
        )

        expected_mlp_params_numel = (
            numel_params_trunk + numel_params_opacity + numel_params_color
        )
        assert expected_mlp_params_numel == mlp_params.numel(), (
            f"The number of elements in mlp param should be {expected_mlp_params_numel}."
            f" Got {mlp_params.numel()} instead."
        )
        # make sure grid_idx is in the correct range
        assert grid_idx.min() >= 0, f"Negative grid index: {grid_idx.min()}"
        assert grid_idx.max() <= (
            grid_batch_size - 1
        ), f"A grid index is out of bounds ({grid_idx.max()} >= {grid_batch_size})"

        # init output tensors
        negative_log_transmittance = torch.zeros(
            num_rays, device=device, dtype=torch.float32
        )
        ray_length_render = torch.zeros(num_rays, device=device, dtype=torch.float32)
        feature_render = torch.zeros(
            num_rays, num_render_channels, device=device, dtype=torch.float32
        )

        # use voxel grid scaffold
        use_scaffold = scaffold is not None
        if scaffold is not None:
            scaffold_t = scaffold.reshape(grid_batch_size, -1, 1).float()
            feature_grid_sizes = torch.cat(
                [
                    feature_grid_sizes,
                    torch.tensor(
                        [[*scaffold.shape, 1]], dtype=torch.int32, device=device
                    ),
                ],
                dim=0,
            ).int()
        else:
            scaffold_t = feature_grid.new_empty(1)

        # Random noise seed for each ray, we have to pass this in as a
        # tensor otherwise triton would jit-recompile every kernel run
        # with a different seed.
        inject_noise = inject_noise_sigma > 0.0
        inject_noise_seed_t = torch.full(
            (num_rays,),
            inject_noise_seed,
            device=device,
            dtype=torch.long,
        )

        n_blocks = int(math.ceil(num_rays / BLOCK_SIZE))
        grid = (n_blocks,)
        fw_kernel[grid](
            # ---- output -----
            _contiguous(negative_log_transmittance),
            _contiguous(ray_length_render),
            _contiguous(feature_render),
            # ---- grid ----
            _contiguous(feature_grid),
            _contiguous(feature_grid_sizes),
            _contiguous(color_feature_grid),
            _contiguous(color_feature_grid_sizes),
            # ----- non-differentiable tensors
            _contiguous(directions),
            _contiguous(origins),
            _contiguous(grid_idx),
            _contiguous(near),
            _contiguous(far),
            _contiguous(ray_encoding),
            _contiguous(inject_noise_seed_t),
            _contiguous(scaffold_t),
            # ---- mlp params ----
            _contiguous(mlp_params),  # master ptr for the mlp params
            mlp_dim_hidden_trunk,
            mlp_dim_hidden_opacity,
            mlp_dim_hidden_color,
            dim_in_trunk,
            dim_in_opacity,
            dim_in_color,
            dim_out_trunk,
            dim_out_color,
            # ----- config keys ----
            num_samples,
            num_samples_inf,
            gain,
            # ----- sizes ----
            num_rays,
            num_grid_channels,
            num_grids,
            num_color_grids,
            BLOCK_SIZE,
            # ---- switches ----
            int(mask_out_of_bounds_samples),
            int(inject_noise),
            float(inject_noise_sigma),
            int(contract_coords),
            float(disparity_at_inf),
            int(use_scaffold),
            int(use_separate_color_grid),
            num_warps=1 if DEBUG else NUM_WARPS,
        )

        # save tensors for bw
        ctx.save_for_backward(
            negative_log_transmittance,
            feature_grid,
            feature_grid_sizes,
            color_feature_grid,
            color_feature_grid_sizes,
            mlp_params,
            directions,
            origins,
            grid_idx,
            near,
            far,
            ray_encoding,
            inject_noise_seed_t,
            scaffold_t,
        )

        # save config keys
        ctx.mlp_dim_hidden_trunk = mlp_dim_hidden_trunk
        ctx.mlp_dim_hidden_opacity = mlp_dim_hidden_opacity
        ctx.mlp_dim_hidden_color = mlp_dim_hidden_color
        ctx.mlp_n_layers_trunk = mlp_n_layers_trunk
        ctx.mlp_n_layers_opacity = mlp_n_layers_opacity
        ctx.mlp_n_layers_color = mlp_n_layers_color
        ctx.dim_in_opacity = dim_in_opacity
        ctx.dim_in_color = dim_in_color
        ctx.dim_out_trunk = dim_out_trunk
        ctx.dim_out_color = dim_out_color
        ctx.num_samples = num_samples
        ctx.num_samples_inf = num_samples_inf
        ctx.gain = gain
        ctx.num_render_channels = num_render_channels
        ctx.mask_out_of_bounds_samples = mask_out_of_bounds_samples
        ctx.contract_coords = contract_coords
        ctx.disparity_at_inf = disparity_at_inf
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.NUM_WARPS = NUM_WARPS
        ctx.num_grids = num_grids
        ctx.num_color_grids = num_color_grids
        ctx.num_rays = num_rays
        ctx.num_grid_channels = num_grid_channels
        ctx.scaffold_t = scaffold_t
        ctx.inject_noise_seed_t = inject_noise_seed_t
        ctx.use_scaffold = use_scaffold
        ctx.use_separate_color_grid = use_separate_color_grid
        ctx.inject_noise = inject_noise
        ctx.inject_noise_sigma = inject_noise_sigma

        if PROFILE:
            torch.cuda.synchronize()
            elapsed = time.time() - time_start
            print(f"fw time = {elapsed:1.5f}")

        return ray_length_render, negative_log_transmittance, feature_render

    @staticmethod
    def backward(
        ctx,
        grad_ray_length_render,
        grad_negative_log_transmittances,
        grad_feature_render,
    ):
        if PROFILE:
            torch.cuda.synchronize()
            time_start = time.time()

        (
            negative_log_transmittances,
            feature_grid,
            feature_grid_sizes,
            color_feature_grid,
            color_feature_grid_sizes,
            mlp_params,
            directions,
            origins,
            grid_idx,
            near,
            far,
            ray_encoding,
            inject_noise_seed_t,
            scaffold_t,
        ) = ctx.saved_tensors

        device = feature_grid.device
        grad_feature_grid = torch.zeros_like(feature_grid)
        grad_mlp_params = torch.zeros_like(mlp_params)
        grad_rays_enc = torch.zeros_like(ray_encoding)

        if ctx.use_separate_color_grid:
            grad_color_feature_grid = torch.zeros_like(color_feature_grid)
        else:
            grad_color_feature_grid = torch.empty(
                (1,), dtype=torch.float32, device=device
            )

        n_blocks = int(math.ceil(ctx.num_rays / ctx.BLOCK_SIZE))
        grid = (n_blocks,)
        debug_tensor = torch.zeros((32, 32)).to(feature_grid.device)

        ctx.bw_kernel[grid](
            negative_log_transmittances,
            # ----- differentiable tensors -----
            _contiguous(feature_grid),
            _contiguous(feature_grid_sizes),
            _contiguous(color_feature_grid),
            _contiguous(color_feature_grid_sizes),
            # ----- non-differentiable tensors -----
            _contiguous(directions),
            _contiguous(origins),
            _contiguous(grid_idx.to(torch.int32)),
            _contiguous(near),
            _contiguous(far),
            _contiguous(ray_encoding),
            _contiguous(inject_noise_seed_t),
            _contiguous(scaffold_t),
            # ----- mlp params -----
            _contiguous(mlp_params),
            ctx.mlp_dim_hidden_trunk,
            ctx.mlp_dim_hidden_opacity,
            ctx.mlp_dim_hidden_color,
            ctx.dim_in_opacity,
            ctx.dim_in_color,
            ctx.dim_out_trunk,
            ctx.dim_out_color,
            # ----- config keys -----
            ctx.num_samples,
            ctx.num_samples_inf,
            ctx.gain,
            # ----- sizes -----
            ctx.num_rays,
            ctx.num_grid_channels,
            ctx.num_grids,
            ctx.num_color_grids,
            ctx.BLOCK_SIZE,
            # ----- switches -----
            int(ctx.mask_out_of_bounds_samples),
            int(ctx.inject_noise),
            ctx.inject_noise_sigma,
            int(ctx.contract_coords),
            ctx.disparity_at_inf,
            int(ctx.use_scaffold),
            int(ctx.use_separate_color_grid),
            # ----- gradients -----
            _contiguous(grad_ray_length_render),
            _contiguous(grad_negative_log_transmittances),
            _contiguous(grad_feature_render),
            # ----- gradients output -----
            _contiguous(grad_feature_grid),
            _contiguous(grad_color_feature_grid),
            _contiguous(grad_mlp_params),
            _contiguous(grad_rays_enc),
            debug_tensor,
            # num_warps=1 if DEBUG else None,
        )

        if PROFILE:
            torch.cuda.synchronize()
            elapsed = time.time() - time_start
            print(f"bw time = {elapsed:1.5f}")

        # TODO: remove for speed
        assert torch.isfinite(grad_feature_grid).all()
        assert torch.isfinite(grad_color_feature_grid).all()
        assert torch.isfinite(grad_mlp_params).all()
        assert torch.isfinite(grad_rays_enc).all()

        return (
            grad_feature_grid,
            None,
            grad_mlp_params,
            None,
            None,
            None,
            None,
            None,
            grad_rays_enc,
            None,
            grad_color_feature_grid if ctx.use_separate_color_grid else None,
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
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _contiguous(t: torch.Tensor | None):
    t_out = t.contiguous() if t is not None else t
    return t_out


def _get_mlp_n_params_weight(dim_in: int, dim_hidden: int, dim_out: int, n_layers: int):
    if n_layers == 0:
        return 0
    if n_layers == 1:
        return dim_in * dim_out
    if n_layers == 2:
        return dim_in * dim_hidden + dim_hidden * dim_out
    # n_layers > 2
    return dim_hidden * dim_hidden * (n_layers - 2) + _get_mlp_n_params_weight(
        dim_in, dim_hidden, dim_out, 2
    )


def _get_mlp_n_params_bias(dim_out: int, dim_hidden: int, n_layers: int):
    return dim_hidden * max(n_layers - 1, 0) + dim_out


def _get_mlp_n_params(dim_in: int, dim_hidden: int, dim_out: int, n_layers: int):
    return _get_mlp_n_params_weight(
        dim_in, dim_hidden, dim_out, n_layers
    ) + _get_mlp_n_params_bias(dim_out, dim_hidden, n_layers)
