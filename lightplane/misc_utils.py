# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch


def assert_shape(x: torch.Tensor, shape: Tuple[int, ...]):
    """
    Helper function to assert the shape of a tensor.

    Args:
        x: Input tensor.
        shape: Expected shape of the input tensor.
    """
    assert x.shape == shape, f"expected shape {shape}, got {x.shape}"


def flatten_grid(grid: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens a list of grids (grid-list) into a single 2D tensor, and return the
    grid sizes.

    Args:
        grid: List of grids to flatten.

    Returns:
        grid_flat: Flattened grid, 2D tensor.
        grid_sizes: Grid sizes, List[List[int]]
    """
    device = grid[0].device
    grid_sizes = torch.stack(
        [torch.tensor(g.shape, dtype=torch.int32, device=device) for g in grid],
        dim=0,
    ).contiguous()
    grid_flat = torch.cat(
        [g.reshape(-1, g.shape[-1]) for g in grid],
        dim=0,
    ).contiguous()
    return grid_flat, grid_sizes


def unflatten_grid(
    grid: torch.Tensor, grid_sizes: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    """
    Unflattens a 2D tensor into a list of grids (grid-list), given the grid sizes.

    Args:
        grid: Flattened grid, 2D tensor.
        grid_sizes: Grid sizes, e.g. `grid_sizes = torch.tensor([[1, 32, 32, 32, 64]])`

    Returns:
        grid_list: List of grids. List[torch.Tensor, ...]
    """
    grid_list_flat = grid.split(
        [grid_size[:-1].prod() for grid_size in grid_sizes],
        dim=0,
    )
    grid_list = [
        grid_flat.reshape(*grid_size)
        for grid_flat, grid_size in zip(grid_list_flat, grid_sizes)
    ]
    return tuple(grid_list)


def if_not_none_else(x: Any, y: Any) -> Any:
    """If x is not None, return x, else return y."""
    return x if x is not None else y


def pad_feature_to_block_size(feature: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Pads the feature to a multiple of block size, similar to pad_to_block_size in
    `Ray` Class.

    Args:
        block_size: Block size to pad to.
    """
    n_rays = feature.shape[0]
    n_blocks = (n_rays + block_size - 1) // block_size
    n_rays_padded = n_blocks * block_size - n_rays

    if n_rays_padded > 0:
        pads = [0] * (feature.ndim * 2)
        pads[-1] = n_rays_padded
        feature = torch.nn.functional.pad(feature, pads, mode="constant", value=0.0)
    return feature


def is_in_bounds(points: torch.Tensor) -> torch.BoolTensor:
    """
    Check if the points are within the bounds of [-1, 1] in all dimensions.
    """
    return (points.abs() <= 1.0).all(-1, keepdim=True)


def _check_list_grid_sizes(grid: List[torch.Tensor], grid_sizes: List[List[int]]):
    """
    Helper function to check if the size of `grid` are consistent with `grid_sizes`.
    Args:
        grid: List of grids.
        grid_sizes: List of grid sizes.
    """
    for i in range(len(grid)):
        assert_shape(grid[i], tuple(grid_sizes[i]))


def check_grid(
    grid: List[torch.Tensor] | torch.Tensor,
    grid_sizes: Optional[List[List[int]]] = None,
):
    """
    Helper function to check the shape of `grid` and `grid_sizes`.

    If `grid` is a list, it checks if the shape of each grid in the list is the
    same as the corresponding shape in `grid_sizes`, if `grid_sizes` is not None.

    If `grid` is a 2D tensor, it checks (1) if the `grid_sizes` are specified, and
    (2) total number of elements in `grid` is equal to the sum of the product of
    each grid size in `grid_sizes`.
    """
    if isinstance(grid, list):
        if grid_sizes is not None:
            _check_list_grid_sizes(grid, grid_sizes)
    elif isinstance(grid, torch.Tensor):
        assert grid_sizes is not None, "grid_sizes cannot be None when grid is a tensor"
        assert (
            sum([np.prod(gs) for gs in grid_sizes]) == grid.numel()
        ), "grid_sizes has to be compatible to grid tensor shapes!"
    else:
        raise NotImplementedError("grid should be either tensor or list")

    return grid, grid_sizes


def check_grid_and_color_grid(
    grid: List[torch.Tensor] | torch.Tensor,
    color_grid: List[torch.Tensor] | torch.Tensor | None,
    grid_sizes: Optional[List[List[int]]] = None,
    color_grid_sizes: Optional[List[List[int]]] = None,
):
    """
    Helper function to check the shape of `grid` and `grid_sizes`, as well as `color_grid`
    and `color_grid_sizes`.

    It checks the following:
        1) `grid` and `color_grid` should have the same type if `color_grid` is not `None`.
        2) `grid` and `color_grid` should have the same batch size and feature dimension
           if `grid` is a list.
        3) `grid_sizes` and `color_grid_sizes` should be compatible with the shapes of `grid`
           and `color_grid` respectively.

    """
    if color_grid is not None:
        assert type(grid) == type(
            color_grid
        ), "grid and color_grid should have the same type"
    if isinstance(grid, list):
        if color_grid is not None:
            assert all(
                cg.shape[0] == g.shape[0] for cg, g in zip(color_grid, grid)
            ), "color_grid's batch size should be the same as grid's batch_size"
            assert all(
                cg.shape[-1] == g.shape[-1] for cg, g in zip(color_grid, grid)
            ), "color_grid's feature dimension should be the same as grid's feature dimension"

            if color_grid_sizes is not None:
                _check_list_grid_sizes(color_grid, color_grid_sizes)

        if grid_sizes is not None:
            _check_list_grid_sizes(grid, grid_sizes)

    elif isinstance(grid, torch.Tensor):
        assert grid_sizes is not None, "grid_sizes cannot be None when grid is a tensor"
        assert (
            sum([np.prod(gs) for gs in grid_sizes]) == grid.numel()
        ), "grid_sizes has to be compatible to grid tensor shapes!"

        if color_grid is not None:
            assert (
                color_grid_sizes is not None
            ), "color_grid_sizes cannot be None when color_grid is a tensor"

            assert (
                sum([np.prod(gs) for gs in color_grid_sizes]) == color_grid.numel()
            ), "grid_sizes has to be compatible to grid tensor shapes!"

    else:
        raise NotImplementedError("grid should be either tensor or list")

    return grid, color_grid, grid_sizes, color_grid_sizes


def process_and_flatten_grid(
    grid: List[torch.Tensor] | torch.Tensor,
    color_grid: List[torch.Tensor] | torch.Tensor | None,
    grid_sizes: Optional[List[List[int]]] = None,
    color_grid_sizes: Optional[List[List[int]]] = None,
):
    """
    Helper function to process and flatten the `grid` and `color_grid`.

    If `grid` is a grid-list, it flattens the grid-list into a single 2D tensor,
    and return the grid sizes in the tensor form.

    If `grid` is a tensor, it converts the `grid_size`s into tensor form.

    If `color_grid` is not None, it processes and flattens the `color_grid` in the
    same way as `grid`.
    """
    if isinstance(grid, list):
        if color_grid is not None:
            color_grid, color_grid_sizes = flatten_grid(color_grid)
        else:
            color_grid, color_grid_sizes = None, None

        grid, grid_sizes = flatten_grid(grid)

    elif isinstance(grid, torch.Tensor):
        grid_sizes = torch.tensor(grid_sizes, device=grid.device, dtype=torch.long)
        if color_grid is not None:
            color_grid_sizes = torch.tensor(
                color_grid_sizes, device=grid.device, dtype=torch.long
            )
    else:
        raise NotImplementedError("grid should be flatten either tensor or list")
    return grid, color_grid, grid_sizes, color_grid_sizes
