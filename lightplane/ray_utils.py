# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple, Optional

import math
import torch
import copy

from dataclasses import dataclass, fields, asdict
from enum import Enum
from logging import getLogger


@dataclass
class Rays:
    """
    Dataclass for representing rendering or splatting rays.

    The 3D point `x` along a ray emitted from a 3D coordinate `origin` in along
    a 3D vector `direction` is given by:
    ```
    x = origin + t * direction
    ```
    where `t` is a scalar in range `[near, far]`.
    Note that `direction` does not have to be l2-normalized.

    In order to render multiple scenes given a single batch of rays, each ray
    is associated with an integer index `grid_idx` which specifies the index of
    its coresponding scene.

    Optionally, the object can store an encoding of the rays: `encoding`.
    This can be useful to define a user-specific encoding of the rays,
    e.g. a custom version of the harmonic embedding originally proposed by NeRF. Note
    that the dimensionality of the embedding has to match the number of channels
    accepted by the corresponding MLP of Lightplane Renderer or Splatter.

    Args:
        directions: Tensor of shape `(B, 3)` storing the directions of `B` rays.
        origins: Tensor of shape `(B, 3)` storing the origins of `B` rays.
        grid_idx: 1D Tensor of shape `(B,)` storing an integer index of each ray into its
            corresponding feature grid.
        near: Tensor of shape `(B,)` storing the ray-length at which raymarching starts.
        far: Tensor of shape `(B,)` storing the ray-length at which raymarching ends.
        encoding: Optional Tensor of shape `(B, C)` storing the encoding of each ray.
    """

    directions: torch.Tensor  # B x 3
    origins: torch.Tensor  # B x 3
    grid_idx: torch.Tensor  # B
    near: torch.Tensor  # B
    far: torch.Tensor  # B
    encoding: Optional[torch.Tensor] = None  # B x C

    @property
    def device(self, assert_same_device: bool = False):
        """
        Return the device on which the rays are stored.

        Args:
            assert_same_device: If True, asserts that all tensors are on the same device.

        Returns:
            device: Device on which the rays are stored.
        """
        device = self.directions.device
        if assert_same_device:
            for f in fields(self):
                v = getattr(self, f.name)
                if v is not None and torch.istensor(v):
                    assert v.device == device, (
                        f"{f.name} is on a different device ({str(v.device)},"
                        + f" expected {str(device)})"
                    )
        return device

    def __post_init__(self):
        _validate_rays(
            self.directions,
            self.origins,
            self.grid_idx,
            self.near,
            self.far,
            self.encoding,
        )

    def __getitem__(self, key):
        """
        Select a subset of the Rays object by indexing with `key`.

        Args:
            key: The indexing key.

        Returns:
            A new Rays object holding a subset of the rays given selected by `key`.
        """
        rays_dict = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if v is not None:
                v = v[key]
            rays_dict[field.name] = v
        return type(self)(**rays_dict)

    def pad_to_block_size(self, block_size: int) -> Tuple["Rays", int]:
        """
        Pads the rays to a multiple of block_size.

        Args:
            block_size: Block size to pad to.

        Returns:
            self_padded: Padded rays.
            n_rays_padded: The number of added rays.
        """
        n_rays = self.directions.shape[0]
        n_blocks = (n_rays + block_size - 1) // block_size
        n_rays_padded = n_blocks * block_size - n_rays
        if n_rays_padded > 0:
            rays_dict = {}
            for f in fields(self):
                v = getattr(self, f.name)
                if v is None:
                    rays_dict[f.name] = v
                else:
                    pads = [0] * (v.ndim * 2)
                    pads[-1] = n_rays_padded
                    v_padded = torch.nn.functional.pad(
                        v, pads, mode="constant", value=0.0
                    )
                    rays_dict[f.name] = v_padded
            self_padded = type(self)(**rays_dict)
        else:
            self_padded = self

        return self_padded, n_rays_padded

    def to(self, device, copy: bool = False) -> "Rays":
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
          device: Device (torch.device) for the new object.
          copy: Boolean indicator whether or not to clone self. Default False.

        Returns:
          Pointclouds object.
        """
        if not copy and self.device == device:
            return self

        other = self.clone()
        if self.device == device:
            return other

        other.device = device
        for f in fields(self):
            v = getattr(self, f)
            if v is not None and torch.istensor(v):
                setattr(other, f, v.to(device))
        return other

    def clone(self) -> "Rays":
        """
        Deep copy of a Rays object.

        Returns:
            new Rays object.
        """
        return copy.deepcopy(self)


def calc_harmonic_embedding(directions: torch.Tensor, n_harmonic_functions: int):
    """Calculates harmonic embedding for the given directions.

    Note that the function is strongly inspired by PyTorch3D's implementation:
        https://github.com/facebookresearch/pytorch3d/blob/c292c71c1adb0712c12cf4fa67a7a84ad9b44e5c/pytorch3d/renderer/implicit/harmonic_embedding.py#L12

    Args:
        directions: Ray directions. ... x 3.
        n_harmonic_functions: Number of harmonic functions. If set to 0, the
            function will only return the input directions, otherwise
            returns the input directions concatenated with the harmonic
            embeddings.

    Returns:
        encoding: Harmonic embedding. ... x n_harmonic_functions.
    """
    if n_harmonic_functions == 0:
        # return the input directions
        return directions

    device, dtype = directions.device, directions.dtype
    frequencies = 2.0 ** torch.arange(
        n_harmonic_functions,
        dtype=dtype,
        device=device,
    )
    zero_half_pi = torch.tensor([0.0, 0.5 * torch.pi], device=device, dtype=dtype)
    embed = directions[..., None] * frequencies
    embed = embed[..., None, :, :] + zero_half_pi[..., None, None]
    embed = embed.sin()
    embed = embed.reshape(*directions.shape[:-1], -1)
    return torch.cat([embed, directions], dim=-1)


def calc_harmonic_embedding_dim(n_harmonic_functions: int) -> int:
    """Calculates the dimension of the harmonic embedding."""
    return 3 + 2 * 3 * n_harmonic_functions  # sin, cos for each coordinate


def jitter_near_far(near: torch.Tensor, far: torch.Tensor, num_samples: int):
    """
    Jitters the near and far planes by a random offset in range [-delta, delta],
    where delta = (far - near) / num_samples.
    """
    delta = (far - near) / num_samples
    offs = (2 * torch.rand_like(near) - 1) * delta
    near = near + offs
    far = far + offs
    return near, far


def _validate_rays(
    directions: torch.Tensor,
    origins: torch.Tensor,
    grid_idx: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    encoding: Optional[torch.Tensor],
):
    """
    Validates the Rays object.

    Args:
        directions: Ray directions. B x 3.
        origins: Ray origins. B x 3.
        grid_idx: Integer index of each ray into its corresponding grid in the batch. B.
        near: Near plane distances. B.
        far: Far plane distances. B.
        encoding: Optional encoding of the rays. B x C.
    """
    n_rays = directions.shape[0]
    assert directions.ndim == 2
    assert origins.ndim == 2
    assert grid_idx.ndim == 1
    assert near.ndim == 1
    assert far.ndim == 1
    assert not grid_idx.is_floating_point()
    assert directions.shape[1] == origins.shape[1] == 3
    device = directions.device
    for vn, v in zip(
        ["directions", "origins", "near", "far", "grid_idx"],
        [directions, origins, near, far, grid_idx],
    ):
        assert (
            v.device == device
        ), f"{vn} is on a wrong device ({str(v.device)}, expected {str(device)})"
        assert (
            v.shape[0] == n_rays
        ), f"Unexpected number of elements in {vn} ({v.shape[0]}, expected {n_rays})"

    if encoding is not None:
        assert encoding.ndim == 2
        assert encoding.shape[0] == n_rays
        assert encoding.device == device
