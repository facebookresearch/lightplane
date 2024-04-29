# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


from .lightplane_splatter import lightplane_mlp_splatter, lightplane_splatter
from .misc_utils import if_not_none_else, unflatten_grid
from .mlp_utils import SplatterParams, init_splatter_params
from .naive_splatter import lightplane_splatter_naive
from .ray_utils import Rays, jitter_near_far


class LightplaneSplatter(torch.nn.Module):
    def __init__(
        self,
        num_samples: int,
        grid_chn: int,
        num_samples_inf: int = 0,
        mask_out_of_bounds_samples: bool = False,
        contract_coords: bool = False,
        disparity_at_inf: float = 1e-5,
        rays_jitter_near_far: bool = False,
        triton_block_size: int = 16,
        triton_num_warps: int = 4,
        use_naive_impl: bool = False,
    ):
        r"""
        This is the Pytorch Module for the Lightplane Splatter.
        It uses `lightplane_splatter` as the core function and directly splats
        `rays.encoding` to a zero-initialized  output grid-list, `output_grid`.

        Args:
            num_samples: Number of samples to splat.
            grid_chn: Number of channels in the `output_grid`.
            num_samples_inf: Number of samples beyond the  `far` plane.
            mask_out_of_bounds_samples: Whether to mask out-of-bounds samples.
            contract_coords: Whether to contract the coordinates as in MeRF
            disparity_at_inf: The beyond-far samples (their number-per-ray is
                determined by `num_samples_inf`) are sampled in the disparity
                space in `range[far, 1 / disparity_at_inf]`.
            rays_jitter_near_far: Whether to jitter the `near` and `far` planes
                uniformly in range `[-delta, delta]`.
            triton_block_size: Block size for triton.
            triton_num_warps: Number of warps for triton.
            use_naive_impl: Whether to use the naive pytorch implementation.
        """
        super().__init__()

        self.num_samples = num_samples
        self.num_samples_inf = num_samples_inf
        self.mask_out_of_bounds_samples = mask_out_of_bounds_samples
        self.contract_coords = contract_coords
        self.disparity_at_inf = disparity_at_inf
        self.rays_jitter_near_far = rays_jitter_near_far
        self.triton_block_size = triton_block_size
        self.triton_num_warps = triton_num_warps
        self.use_naive_impl = use_naive_impl

        self.rays_encoding_dim = grid_chn

    def get_splatter_params(self) -> SplatterParams:
        r"""
        Helper function to get the splatter parameters.
        """
        return None

    def forward(
        self,
        rays: Rays,
        grid_size: list[tuple[int, int, int, int, int]],
        # If set, the following args override the module's default values:
        num_samples: int | None = None,
        num_samples_inf: int | None = None,
        mask_out_of_bounds_samples: bool | None = None,
        contract_coords: bool | None = None,
        disparity_at_inf: float | None = None,
        rays_jitter_near_far: bool | None = None,
        return_list: bool = True,  # return grid list instead of a stacked tensor
        regenerate_code: bool = False,
    ):
        r"""
        Forward function for splatting rays into a 'output_grid'.

        Args:
            rays: `Rays` to splat. `rays.encoding` is splatted to the `output_grid`.
            grid_size: List of tuples specifying the grid sizes of `output_grid`.
            num_samples: Number of samples for splatting.
            num_samples_inf: Number of samples beyond the `far` plane.
            mask_out_of_bounds_samples: Whether to mask out-of-bounds samples.
            contract_coords: Whether to contract the coordinates as in MeRF.
            disparity_at_inf: The beyond-far samples (their number-per-ray is
                determined by `num_samples_inf`) are sampled in the disparity
                space in `range[far, 1 / disparity_at_inf]`.
            rays_jitter_near_far: Whether to jitter the `near` and `far` planes
                uniformly in range `[-delta, delta]`.
            return_list: Whether to return a list of grids or a stacked tensor.
            regenerate_code: Whether to regenerate the code for the splatter.
        Returns:
            output_grid: splatted grid.
        """
        num_samples = if_not_none_else(num_samples, self.num_samples)
        num_samples_inf = if_not_none_else(num_samples_inf, self.num_samples_inf)
        mask_out_of_bounds_samples = if_not_none_else(
            mask_out_of_bounds_samples,
            self.mask_out_of_bounds_samples,
        )
        contract_coords = if_not_none_else(contract_coords, self.contract_coords)
        disparity_at_inf = if_not_none_else(disparity_at_inf, self.disparity_at_inf)
        rays_jitter_near_far = if_not_none_else(
            rays_jitter_near_far, self.rays_jitter_near_far
        )

        # handle ray encoding
        _check_splatter_ray_encoding_input(
            rays.encoding,
            self.rays_encoding_dim,
        )

        rays = copy.copy(rays)  # shallow copy for futher processing

        if rays_jitter_near_far:
            rays.near, rays.far = jitter_near_far(
                rays.near,
                rays.far,
                num_samples,
            )

        kwargs = {
            "rays": rays,
            "output_grid_size": grid_size,
            "num_samples": num_samples,
            "num_samples_inf": num_samples_inf,
            "mask_out_of_bounds_samples": mask_out_of_bounds_samples,
            "contract_coords": contract_coords,
            "disparity_at_inf": disparity_at_inf,
            "return_list": return_list,
            "triton_block_size": self.triton_block_size,
            "triton_num_warps": self.triton_num_warps,
            "regenerate_code": regenerate_code,
        }
        # run lightplane
        if self.use_naive_impl:
            lightplane_fn = lightplane_splatter_naive
        else:
            lightplane_fn = lightplane_splatter

        out = lightplane_fn(**kwargs)

        return out


class LightplaneMLPSplatter(torch.nn.Module):
    def __init__(
        self,
        num_samples: int,
        grid_chn: int,
        input_grid_chn: int = 32,
        mlp_hidden_chn: int = 32,
        mlp_n_layers: int = 2,
        num_samples_inf: int = 0,
        mask_out_of_bounds_samples: bool = False,
        contract_coords: bool = False,
        disparity_at_inf: float = 1e-5,
        rays_jitter_near_far: bool = False,
        triton_block_size: int = 16,
        triton_num_warps: int = 4,
        use_naive_impl: bool = False,
    ):
        r"""
        This is the Pytorch Module for the Lightplane Splatter.
        It uses `lightplane_mlp_splatter` as the core function and samples the
        point feature from the corresponding prior input grid `input_grid`, adds
        the sampled feature to the `encoding` of the ray, passes the latter
        through an MLP, and splats the MLP output to the grid-list `output_grid`.

        Args:
            num_samples: Number of samples to splat.
            grid_chn: Number of channels in the `output_grid`.
            input_grid_chn: Number of channels in the `input_grid`.
                It should be the same as the number of channels for `rays.encoding`.
            mlp_hidden_chn: Number of hidden channels in the MLP.
            mlp_n_layers: Number of layers in the MLP.
            num_samples_inf: Number of samples beyond the `far` plane.
            mask_out_of_bounds_samples: Whether to mask out-of-bounds samples.
            contract_coords: Whether to contract the coordinates as in MeRF
            disparity_at_inf: The beyond-far samples (their number-per-ray is
                determined by `num_samples_inf`) are sampled in the disparity
                space in `range[far, 1 / disparity_at_inf]`.
            rays_jitter_near_far: Whether to jitter the `near` and `far` planes
                uniformly in range `[-delta, delta]`.
            triton_block_size: Block size for triton.
            triton_num_warps: Number of warps for triton.
            use_naive_impl: Whether to use the naive pytorch implementation.
        """

        super().__init__()

        self.num_samples = num_samples
        self.num_samples_inf = num_samples_inf
        self.mask_out_of_bounds_samples = mask_out_of_bounds_samples
        self.contract_coords = contract_coords
        self.disparity_at_inf = disparity_at_inf
        self.rays_jitter_near_far = rays_jitter_near_far
        self.triton_block_size = triton_block_size
        self.triton_num_warps = triton_num_warps
        self.use_naive_impl = use_naive_impl

        assert input_grid_chn is not None, "input_grid_chn must be provided"
        splatter_params = init_splatter_params(
            device="cpu",
            n_layers=mlp_n_layers,
            input_chn=input_grid_chn,
            hidden_chn=mlp_hidden_chn,
            out_chn=grid_chn,
        )
        self.mlp_params = torch.nn.Parameter(splatter_params.mlp_params)
        # register the n_hidden field of decoder_params
        self.register_buffer("n_hidden", splatter_params.n_hidden, persistent=False)
        # ray encoding dim is the same as the output channels of the mlp
        self.rays_encoding_dim = input_grid_chn

    def get_splatter_params(self) -> SplatterParams:
        r"""
        Helper function to get the splatter parameters.
        """
        return SplatterParams(self.mlp_params, self.n_hidden)

    def forward(
        self,
        rays: Rays,
        grid_size: list[tuple[int, int, int, int, int]],
        input_grid: tuple[torch.Tensor, ...] | torch.Tensor,
        num_samples: int | None = None,
        num_samples_inf: int | None = None,
        mask_out_of_bounds_samples: bool | None = None,
        contract_coords: bool | None = None,
        disparity_at_inf: float | None = None,
        input_grid_sizes: list[list[int]] | None = None,
        rays_jitter_near_far: bool | None = None,
        return_list: bool = True,  # return grid list instead of a stacked tensor
        regenerate_code: bool = False,
    ):
        r"""
        Forward function for splatting rays into a 'output_grid' with an MLP and
        `input_grid` as the prior grid.

        Args:
            rays: `Rays` to splat. `rays.encoding` is splatted to the `output_grid`.
            grid_size: List of tuples specifying the grid sizes of `output_grid`.
            input_grid:  Grids to sample the point feature from.
            num_samples: Number of samples for splatting.
            num_samples_inf: Number of samples beyond the `far` plane.
            mask_out_of_bounds_samples: Whether to mask out-of-bounds samples.
            contract_coords: Whether to contract the coordinates as in MeRF.
            disparity_at_inf: The beyond-far samples (their number-per-ray is
                determined by `num_samples_inf`) are sampled in the disparity
                space in `range[far, 1 / disparity_at_inf]`.
            input_grid_sizes: The size of the `input_grid`. Only required if
                `input_grid` is a 2D tensor.
            rays_jitter_near_far: Whether to jitter the `near` and `far` planes
                uniformly in range `[-delta, delta]`.
            return_list: Whether to return a list of grids or a stacked tensor.
            regenerate_code: Whether to regenerate the code for the splatter.
        Returns:
            output_grid: splatted grid.
        """
        num_samples = if_not_none_else(num_samples, self.num_samples)
        num_samples_inf = if_not_none_else(num_samples_inf, self.num_samples_inf)
        mask_out_of_bounds_samples = if_not_none_else(
            mask_out_of_bounds_samples,
            self.mask_out_of_bounds_samples,
        )
        contract_coords = if_not_none_else(contract_coords, self.contract_coords)
        disparity_at_inf = if_not_none_else(disparity_at_inf, self.disparity_at_inf)
        rays_jitter_near_far = if_not_none_else(
            rays_jitter_near_far, self.rays_jitter_near_far
        )

        # handle ray encoding
        _check_splatter_ray_encoding_input(
            rays.encoding,
            self.rays_encoding_dim,
        )

        assert input_grid is not None, "input_grid must be provided"

        rays = copy.copy(rays)  # shallow copy for futher processing

        if rays_jitter_near_far:
            rays.near, rays.far = jitter_near_far(
                rays.near,
                rays.far,
                num_samples,
            )

        kwargs = {
            "rays": rays,
            "output_grid_size": grid_size,
            "mlp_params": self.get_splatter_params(),
            "input_grid": input_grid,
            "num_samples": num_samples,
            "num_samples_inf": num_samples_inf,
            "mask_out_of_bounds_samples": mask_out_of_bounds_samples,
            "contract_coords": contract_coords,
            "disparity_at_inf": disparity_at_inf,
            "return_list": return_list,
            "triton_block_size": self.triton_block_size,
            "triton_num_warps": self.triton_num_warps,
            "regenerate_code": regenerate_code,
        }
        # run lightplane
        if self.use_naive_impl:
            lightplane_fn = lightplane_splatter_naive
        else:
            lightplane_fn = lightplane_mlp_splatter

        out = lightplane_fn(**kwargs)

        return out


def _check_splatter_ray_encoding_input(
    ray_encoding: torch.Tensor | None,
    ray_encoding_dim: int,
):
    if ray_encoding is None:
        raise ValueError(
            "The encoding field of input rays is None."
            " However, the Splatter requires an encoding for input rays."
        )

    if ray_encoding is not None and ray_encoding.shape[1] != ray_encoding_dim:
        raise ValueError(
            f"Ray encoding has a wrong dimension."
            f" Expected: {ray_encoding_dim}, got: {ray_encoding.shape[1]}"
        )
