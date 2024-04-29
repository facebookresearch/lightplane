# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
from typing import List, Optional, Tuple

import numpy as np
import torch


def init_3d_representation(
    representation_type: str,
    grid_resolution: int,
    num_grid_channels: int,
    device: torch.device,
):
    if representation_type == "voxel_grid":
        r = grid_resolution
        v = torch.rand(1, r, r, r, num_grid_channels, device=device)
        v.requires_grad = True
        return [v]

    elif representation_type == "triplane":
        grid = []
        for planei in range(3):
            r = grid_resolution
            size = [1, r, r, r, num_grid_channels]
            size[planei + 1] = 1
            triplane = (4 / pow(num_grid_channels, 1 / 3)) * torch.rand(
                size, device=device
            )
            triplane.requires_grad = True
            grid.append(triplane)
        return grid

    raise ValueError(f"Unsupported representation {representation_type}.")


def random_grid(
    size: Tuple[int, ...],
    device: torch.device,
    requires_grad: bool,
    is_triplane: bool,
):
    if is_triplane:
        grid = []
        for i in range(3):
            size_ = list(copy.deepcopy(size))
            size_[i + 1] = 1
            grid_ = torch.randn(size_, device=device, dtype=torch.float32)
            grid_.requires_grad = requires_grad
            grid.append(grid_)
    else:
        grid = torch.randn(size, device=device, dtype=torch.float32)
        # grid = torch.randint(0, 10, size, device=device, dtype=torch.float32)
        # grid = torch.ones(size, device=device, dtype=torch.float32)
        grid.requires_grad = requires_grad
        grid = [grid]
    return grid


def _grid_l1_loss(grid: torch.Tensor) -> torch.Tensor:
    """
    Help function to calculate L1 loss on a grid structure.
    grid struture has the shape B x D x W x H x C, (D, W, H could be 1 for planes).
    """
    assert grid.ndim == 5
    return torch.mean(torch.abs(grid))


def _grid_tv_loss(grid: torch.Tensor) -> torch.Tensor:
    """
    Help function to calculate TV loss on a grid structure.
    grid struture has the shape B x D x W x H x C, (D, W, H could be 1 for planes).
    """
    assert grid.ndim == 5
    n_non_singular_dim = sum(int(s > 1) for s in grid.shape[1:-1])

    if n_non_singular_dim == 3:  # 3d voxel grid
        batch_size = grid.size()[0]
        d_x = grid.size()[1]
        h_x = grid.size()[2]
        w_x = grid.size()[3]

        count_d = np.prod(grid[:, 1:, :, :].shape[1:])
        count_h = np.prod(grid[:, :, 1:, :].shape[1:])
        count_w = np.prod(grid[:, :, :, 1:].shape[1:])

        d_tv = torch.pow((grid[:, 1:, :, :] - grid[:, : d_x - 1, :, :]), 2).sum()
        h_tv = torch.pow((grid[:, :, 1:, :] - grid[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((grid[:, :, :, 1:] - grid[:, :, :, : w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w + d_tv / count_d) / batch_size
    elif n_non_singular_dim == 2:  # triplane
        singular_dim = [i for i, s in enumerate(grid.shape[1:-1]) if s == 1][0]
        grid = grid.squeeze(singular_dim + 1)
        batch_size = grid.size()[0]
        h_x = grid.size()[1]
        w_x = grid.size()[2]

        count_h = np.prod(grid[:, 1:, :, :].shape[1:])
        count_w = np.prod(grid[:, :, 1:, :].shape[1:])

        h_tv = torch.pow((grid[:, 1:, :] - grid[:, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((grid[:, :, 1:] - grid[:, :, : w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w + w_tv / count_w) / batch_size
    else:
        raise NotImplementedError("Non Singular dim error.")


@torch.no_grad()
def _up_sample_grid(
    grid: torch.Tensor, upsample_factor: float, align_corners: bool
) -> torch.Tensor:
    """
    Help function to upsample grid structures.
    grid struture has the shape B x D x W x H x C, (D, W, H could be 1 for planes).
    """
    assert grid.ndim == 5
    n_non_singular_dim = sum(int(s > 1) for s in grid.shape[1:-1])

    if n_non_singular_dim == 3:  # 3d voxel grid
        grid = grid.permute(0, 4, 1, 2, 3)
        grid = torch.nn.functional.interpolate(
            grid,
            scale_factor=upsample_factor,
            mode="trilinear",
            align_corners=align_corners,
        )
        grid = grid.permute(0, 2, 3, 4, 1).contiguous()
        grid.requires_grad = True
        return grid
    elif n_non_singular_dim == 2:  # triplane
        singular_dim = [i for i, s in enumerate(grid.shape[1:-1]) if s == 1][0]
        grid = grid.squeeze(singular_dim + 1).permute(0, 3, 1, 2)
        grid = (
            torch.nn.functional.interpolate(
                grid,
                scale_factor=upsample_factor,
                mode="bilinear",
                align_corners=align_corners,
            )
            .permute(0, 2, 3, 1)
            .unsqueeze(singular_dim + 1)
        ).contiguous()
        grid.requires_grad = True
        return grid
    else:
        raise NotImplementedError("Non Singular dim error.")


def grid_TV_loss(
    grids: List[torch.Tensor],
    color_grid: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    tv_loss = 0
    for _grid in grids:
        tv_loss += _grid_tv_loss(_grid)
    if color_grid is not None:
        for _grid in color_grid:
            tv_loss += _grid_tv_loss(_grid)
    return tv_loss


def grid_L1_loss(
    grids: List[torch.Tensor],
    color_grid: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    l1_loss = 0
    for _grid in grids:
        l1_loss += _grid_l1_loss(_grid)
    if color_grid is not None:
        for _grid in color_grid:
            l1_loss += _grid_l1_loss(_grid)
    return l1_loss


@torch.no_grad()
def grid_up_sample(
    grids: List[torch.Tensor],
    upsample_factor: float = 2.0,
    align_corners: bool = False,
):
    for idx in range(len(grids)):
        grids[idx] = _up_sample_grid(grids[idx], upsample_factor, align_corners)
    return grids
