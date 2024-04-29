# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from lightplane import LightplaneRenderer, Rays

from .camera_util import get_rays_for_extrinsic


def get_predicton_for_view(
    render_module: LightplaneRenderer,
    image_size: int,
    near: float,
    far: float,
    R: torch.Tensor,
    T: torch.Tensor,
    grid: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = R.device
    n_images = R.shape[0]
    n_rays_per_image = image_size * image_size
    near_t = torch.ones(n_images, image_size, image_size, device=device) * near
    far_t = torch.ones(n_images, image_size, image_size, device=device) * far
    rays, centers = get_rays_for_extrinsic(image_size, R, T)
    grid_idx = torch.zeros(  # all samples come from a single grid
        n_images * n_rays_per_image,
        device=device,
        dtype=torch.int32,
    )
    return render_module(
        rays=Rays(
            directions=rays.reshape(-1, 3).contiguous(),
            origins=centers.reshape(-1, 3).contiguous(),
            grid_idx=grid_idx.contiguous(),
            near=near_t.reshape(-1).contiguous(),
            far=far_t.reshape(-1).contiguous(),
        ),
        feature_grid=grid,
    )
