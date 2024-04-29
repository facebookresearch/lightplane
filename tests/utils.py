# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy
import dataclasses
import random
import time
from typing import List, Optional, Tuple, Union

import numpy
import numpy as np
import torch

from lightplane import (
    DecoderParams,
    Rays,
    SplatterParams,
    init_decoder_params,
    init_splatter_params,
    lightplane_mlp_splatter,
    lightplane_splatter,
)
from lightplane.naive_splatter import (
    lightplane_mlp_splatter_naive,
    lightplane_splatter_naive,
)


class Memory:
    # memory returned in MB
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.base_memory = torch.cuda.memory_allocated()  # maybe after sync?
        self.base_memory_max = torch.cuda.max_memory_allocated()
        return self

    def get_stats_dict(self):
        return self.stats

    def __exit__(self, type, value, traceback):
        torch.cuda.synchronize()
        memory_after_fw = torch.cuda.memory_allocated()
        memory_max_after_fw = torch.cuda.max_memory_allocated()
        fw_memory_usage = (memory_after_fw - self.base_memory) / 1024.0
        fw_max_memory_usage = (memory_max_after_fw - self.base_memory_max) / 1024.0
        self.stats = {
            f"mem_{self.name}": fw_memory_usage / 1000,
            f"max_mem_{self.name}": fw_max_memory_usage / 1000,
        }


class Timer:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.time()
        self.stats = {}
        return self

    def get_stats_dict(self):
        return self.stats

    def __exit__(self, type, value, traceback):
        torch.cuda.synchronize()
        elapsed = time.time() - self.start
        self.stats = {
            f"t_{self.name}": elapsed,
        }


def lightplane_splatter_test_warpper(
    rays: Rays,
    output_grid_size: list[tuple[int, int, int, int, int]],
    mlp_params: SplatterParams | None = None,
    input_grid: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
    # ------ config keys ------
    num_samples: int = 128,
    num_samples_inf: int = 0,
    mask_out_of_bounds_samples: bool = False,
    contract_coords: bool = False,
    disparity_at_inf: float = 1e-5,
    input_grid_sizes: list[list[int]] | None = None,
    return_list: bool = True,  # whether return list or flatten tensor
    regenerate_code: bool = False,
    triton_block_size: int = 16,
    triton_num_warps: int = 4,
    use_naive_implementation: bool = False,
    checkpointing: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
    """
    This a a warp function for testing lightplane splatter
    (lightplane_mlp_splatter, lightplane_mlp_splatter_naive,
    lightplane_splatter, lightplane_splatter_naive),
    since these functions have different input arguments.
    """
    if input_grid is None:
        if checkpointing is False:
            fn = (
                lightplane_splatter_naive
                if use_naive_implementation
                else lightplane_splatter
            )
            output_grid = fn(
                rays=rays,
                output_grid_size=output_grid_size,
                num_samples=num_samples,
                num_samples_inf=num_samples_inf,
                mask_out_of_bounds_samples=mask_out_of_bounds_samples,
                contract_coords=contract_coords,
                disparity_at_inf=disparity_at_inf,
                return_list=return_list,
                regenerate_code=regenerate_code,
                triton_block_size=triton_block_size,
                triton_num_warps=triton_num_warps,
            )
        else:
            output_grid = lightplane_splatter_naive(
                rays=rays,
                output_grid_size=output_grid_size,
                num_samples=num_samples,
                num_samples_inf=num_samples_inf,
                mask_out_of_bounds_samples=mask_out_of_bounds_samples,
                contract_coords=contract_coords,
                disparity_at_inf=disparity_at_inf,
                return_list=return_list,
                regenerate_code=regenerate_code,
                triton_block_size=triton_block_size,
                triton_num_warps=triton_num_warps,
                checkpointing=True,
            )
    else:
        if checkpointing is False:
            fn = (
                lightplane_mlp_splatter_naive
                if use_naive_implementation
                else lightplane_mlp_splatter
            )
            output_grid = fn(
                rays=rays,
                output_grid_size=output_grid_size,
                mlp_params=mlp_params,
                input_grid=input_grid,
                num_samples=num_samples,
                num_samples_inf=num_samples_inf,
                mask_out_of_bounds_samples=mask_out_of_bounds_samples,
                contract_coords=contract_coords,
                input_grid_sizes=input_grid_sizes,
                disparity_at_inf=disparity_at_inf,
                return_list=return_list,
                regenerate_code=regenerate_code,
                triton_block_size=triton_block_size,
                triton_num_warps=triton_num_warps,
            )
        else:
            output_grid = lightplane_mlp_splatter_naive(
                rays=rays,
                output_grid_size=output_grid_size,
                mlp_params=mlp_params,
                input_grid=input_grid,
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
                checkpointing=True,
            )
    return output_grid


def compare_one(
    ours: torch.Tensor,
    their: torch.Tensor,
    name: str | None = None,
    run_assert: bool = True,
    verbose: bool = True,
    max_abs_df_thresh: float = 0.1,
    mean_abs_df_thresh: float = 0.002,
    mean_rel_df_thresh: float = 0.0007,
):
    df = (ours - their).abs()
    mu_df = df.mean().item()
    max_df = df.max().item()
    E = their.reshape(-1).abs().mean()
    if name is None:
        name = ""
    rel_df = df / their.abs().clamp(1e-7)
    rel_mu_df = rel_df.mean().item()
    rel_max_df = rel_df.max().item()
    if verbose:
        if name is None:
            name = ""
        print(
            f"{name} max={max_df:1.2} mu={mu_df:1.2} "
            + f"rmax={rel_max_df:1.2} rmu={rel_mu_df:1.2} E={E:1.2}"
        )

    all_good = (
        max_df <= max_abs_df_thresh
        and mu_df <= mean_abs_df_thresh
        and rel_mu_df <= mean_rel_df_thresh
    )

    if run_assert:
        assert all_good

    return all_good


def seed_random_generators(seed: int):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def random_rays(
    tot_n_rays: int,
    batch_size: int,
    device: torch.device,
    ray_enc_dim: int | None,
    requires_grad: bool,
    one_ray: bool,
) -> Rays:
    grid_idx = torch.randint(
        0, batch_size, (tot_n_rays,), device=device, dtype=torch.long
    )

    # grid_idx = torch.linspace(
    #     0, batch_size-1, tot_n_rays, device=device,
    # ).round().long()
    # grid_idx = grid_idx[torch.randperm(grid_idx.numel())]

    def _randn(channels: int):
        if one_ray:
            return torch.randn(1, channels, device=device).repeat(tot_n_rays, 1)
        else:
            return torch.randn(tot_n_rays, channels, device=device)

    origins = _randn(3) / 3.0
    directions = -origins + _randn(3) * 0.1
    near = _randn(1)[:, 0] * 0.1 + 0.1
    far = _randn(1)[:, 0].abs() * 0.1 + 3.0
    encoding = None if ray_enc_dim is None else _randn(ray_enc_dim)
    if encoding is not None:
        encoding.requires_grad = requires_grad

    return Rays(
        directions=directions,
        origins=origins,
        grid_idx=grid_idx,
        near=near,
        far=far,
        encoding=encoding,
    )


def get_test_grid_size(size: tuple[int, ...], is_triplane: bool):
    new_size = []
    if is_triplane:
        for i in range(3):
            size_ = list(copy.deepcopy(size))
            size_[i + 1] = 1
            new_size.append(size_)
    else:
        new_size = [size]
    return new_size


def random_grid(
    size: tuple[int, ...],
    device: torch.device,
    requires_grad: bool,
    is_triplane: bool,
    use_tensor_grid: bool = False,  # return a list of tensors or flatten tensors
):
    if is_triplane:
        if use_tensor_grid:
            size_s = []
            for i in range(3):
                size_ = list(copy.deepcopy(size))
                size_[i + 1] = 1
                size_s.append(size_)

            grid = torch.randn(
                (sum([np.prod(s[:-1]) for s in size_s]), size_s[0][-1]),
                device=device,
                dtype=torch.float32,
            )
            grid.requires_grad = requires_grad
        else:
            grid = []
            for i in range(3):
                size_ = list(copy.deepcopy(size))
                size_[i + 1] = 1
                grid_ = torch.randn(size_, device=device, dtype=torch.float32)
                grid_.requires_grad = requires_grad
                grid.append(grid_)
    else:
        if use_tensor_grid:
            grid = torch.randn(
                (np.prod(size[:-1]), size[-1]), device=device, dtype=torch.float32
            )
            grid.requires_grad = requires_grad
        else:
            grid = torch.randn(size, device=device, dtype=torch.float32)
            # grid = torch.randint(0, 10, size, device=device, dtype=torch.float32)
            # grid = torch.ones(size, device=device, dtype=torch.float32)
            grid.requires_grad = requires_grad
            grid = [grid]
    return grid


def random_mlp_decoder_params(
    device: torch.device,
    n_layers_opacity: int = -1,
    n_layers_trunk: int = -1,
    n_layers_color: int = -1,
    input_chn: int = 32,
    hidden_chn: int = 32,
    color_chn: int = 3,
    requires_grad: bool = False,
    use_separate_color_grid: bool = False,
):
    p = init_decoder_params(
        device=device,
        n_layers_opacity=n_layers_opacity,
        n_layers_trunk=n_layers_trunk,
        n_layers_color=n_layers_color,
        input_chn=input_chn,
        hidden_chn=hidden_chn,
        color_chn=color_chn,
        use_separate_color_grid=use_separate_color_grid,
    )

    p.mlp_params.normal_(0.0, 0.01)

    # for p_ in [
    #     p.biases_trunk,
    #     p.biases_opacity,
    #     p.biases_color,
    #     # p.weights_trunk,
    #     # p.weights_opacity,
    #     # p.weights_color,
    # ]:
    #     for p__ in p_:
    #         p__.normal_() * 0.01

    # if True:
    #     print("\n\n\n!!!! REMOVE THIS !!!!\n\n\n")
    #     for p_ in [
    #         p.biases_trunk, p.biases_opacity, p.biases_color,
    #         # p.weights_trunk, p.weights_opacity, p.weights_color,
    #     ]:
    #         for p__ in p_:
    #             p__.fill_(0.0)

    if requires_grad:
        for _, val in p.__dict__.items():
            if torch.is_tensor(val) and torch.is_floating_point(val):
                val.requires_grad = True

    return p


def random_single_mlp_params(
    device: torch.device,
    n_layers: int = -1,
    input_chn: int = 32,
    hidden_chn: int = 32,
    out_chn: int = 3,
    requires_grad: bool = False,
):
    p = init_splatter_params(
        device=device,
        n_layers=n_layers,
        input_chn=input_chn,
        hidden_chn=hidden_chn,
        out_chn=out_chn,
    )

    p.mlp_params.normal_(0.0, 0.01)

    if requires_grad:
        for _, val in p.__dict__.items():
            if torch.is_tensor(val) and torch.is_floating_point(val):
                val.requires_grad = True

    return p


def get_test_example(
    grid_size,
    grid_size_color,
    n_rays,
    n_hidden,
    color_chn,
    n_layers_trunk,
    n_layers_opacity,
    n_layers_color,
    scaffold_size,
    is_triplane: bool,
    requires_grad: bool,
    one_ray: bool = False,
    use_tensor_grid: bool = False,  # whether grid and color_grid are flatten tensors or list of tensors
):
    use_separate_color_grid = grid_size_color is not None
    if use_separate_color_grid:
        ray_enc_dim = grid_size[
            -1
        ]  # the ray encoding is added to sampled color features
    else:
        ray_enc_dim = n_hidden
    device = torch.device("cuda:0")
    B, D, H, W, C = grid_size
    rays = random_rays(
        n_rays,
        B,
        device,
        ray_enc_dim=ray_enc_dim,
        requires_grad=requires_grad,
        one_ray=one_ray,
    )

    mlp_decoder_params = random_mlp_decoder_params(
        device,
        n_layers_trunk=n_layers_trunk,
        n_layers_opacity=n_layers_opacity,
        n_layers_color=n_layers_color,
        input_chn=C,
        hidden_chn=n_hidden,
        color_chn=color_chn,
        requires_grad=requires_grad,
        use_separate_color_grid=use_separate_color_grid,
    )

    grid = random_grid(
        size=grid_size,
        device=device,
        requires_grad=requires_grad,
        is_triplane=is_triplane,
        use_tensor_grid=use_tensor_grid,
    )

    if use_tensor_grid:
        grid_sizes = get_test_grid_size(size=grid_size, is_triplane=is_triplane)
    else:
        grid_sizes = None

    if scaffold_size is not None:
        scaffold = (
            random_grid(
                size=(B, *scaffold_size),
                device=device,
                requires_grad=requires_grad,
                is_triplane=False,
            )[0]
            > 0.0
        ).float()
    else:
        scaffold = None

    if use_separate_color_grid:
        color_grid = random_grid(
            size=[B, *grid_size_color, C],
            device=device,
            requires_grad=requires_grad,
            is_triplane=is_triplane,
            use_tensor_grid=use_tensor_grid,
        )
        if use_tensor_grid:
            color_grid_sizes = get_test_grid_size(
                size=[B, *grid_size_color, C], is_triplane=is_triplane
            )
        else:
            color_grid_sizes = None
    else:
        color_grid = None
        color_grid_sizes = None
    return (
        rays,
        grid,
        grid_sizes,
        color_grid,
        color_grid_sizes,
        mlp_decoder_params,
        scaffold,
    )


def get_test_example_splatter(
    grid_size,
    n_rays,
    feat_dim: int,
    use_mlp: bool,
    n_hidden: int,
    n_layers: int,
    is_triplane: bool,
    requires_grad: bool,
    input_grid_size: list[list[int]] | None = None,
    one_ray: bool = False,
    use_tensor_grid: bool = False,
):
    device = torch.device("cuda:0")
    B, D, H, W, C = grid_size

    output_grid_sizes = []
    if is_triplane:
        output_grid_sizes = []
        input_grid_sizes = []
        for i in range(3):
            feat_grid_ = copy.deepcopy(grid_size)
            input_grid_size_ = copy.deepcopy(input_grid_size)
            feat_grid_[i + 1] = 1
            if input_grid_size_ is not None:
                input_grid_size_[i + 1] = 1
            output_grid_sizes.append(feat_grid_)
            input_grid_sizes.append(input_grid_size_)
    else:
        output_grid_sizes = [grid_size]
        input_grid_sizes = [input_grid_size]

    rays = random_rays(
        n_rays,
        B,
        device,
        ray_enc_dim=None,
        requires_grad=requires_grad,
        one_ray=one_ray,
    )

    splatting_feature = torch.rand(
        (n_rays, feat_dim), device=device, requires_grad=requires_grad
    )
    if use_mlp:
        mlp_params = random_single_mlp_params(
            device,
            n_layers=n_layers,
            input_chn=feat_dim,
            hidden_chn=n_hidden,
            out_chn=C,
            requires_grad=requires_grad,
        )

        if use_tensor_grid:
            input_grid = torch.randn(
                (
                    sum([np.prod(grid_size[:-1]) for grid_size in input_grid_sizes]),
                    input_grid_sizes[0][-1],
                ),
                device=device,
                dtype=torch.float32,
            )
        else:
            input_grid = []
            for size_ in input_grid_sizes:
                grid_ = torch.randn(size_, device=device, dtype=torch.float32)
                grid_.requires_grad = requires_grad
                input_grid.append(grid_)
            input_grid_sizes = None

    else:
        mlp_params = None
        input_grid = None
        input_grid_sizes = None

    return (
        rays,
        output_grid_sizes,
        splatting_feature,
        mlp_params,
        input_grid,
        input_grid_sizes,
    )
