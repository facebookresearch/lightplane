# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math

import numpy
import pytorch3d
import pytorch3d.renderer
import torch
import torch.nn.functional as F

from lightplane import LightplaneRenderer, Rays


def get_sphere_cameras(n_cameras, elevation=30, distance=2.5, device=torch.device("cpu")):
    R = []
    T = []

    for angle in torch.linspace(0, 360, n_cameras + 1)[:n_cameras]:
        R_, T_ = pytorch3d.renderer.look_at_view_transform(distance, elevation, angle)
        R.append(R_[0].to(device))
        T.append(T_[0].to(device))
    return torch.stack(R), torch.stack(T)


def get_rays_for_extrinsic(image_size, R, T):
    device = R.device
    batch_size = R.shape[0]
    half_pixel = 1 / image_size
    canonical_rays = torch.stack(
        torch.meshgrid(
            torch.linspace(1 - half_pixel, -1 + half_pixel, image_size, device=device),
            torch.linspace(1 - half_pixel, -1 + half_pixel, image_size, device=device),
            indexing="xy",
        ),
        dim=-1,
    )
    canonical_rays = torch.nn.functional.pad(canonical_rays, pad=(0, 1), value=1)
    camera = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=R.device)
    rays = camera.unproject_points(
        canonical_rays[None].expand(batch_size, -1, -1, -1).reshape(batch_size, -1, 3)
    )
    center = -(T[:, None] @ R.permute(0, 2, 1))[:, 0]
    centers = center.reshape(-1, 1, 3).expand(-1, image_size * image_size, -1)
    rays = rays - centers
    return rays.to(device), centers.to(device)


def generate_random_cameras(
    batch_size=5,
    distance_range=(1, 10),
    elevation_range=(-30, 30),
    azimuth_range=(0, 360),
    device=torch.device("cpu"),
):
    """
    Generates random camera rotation matrices and translation vectors for a batch of cameras.

    Parameters:
    - batch_size: The number of cameras.
    - distance_range: Tuple (min_distance, max_distance) for camera distances.
    - elevation_range: Tuple (min_elevation, max_elevation) in degrees for camera elevations.
    - azimuth_range: Tuple (min_azimuth, max_azimuth) in degrees for camera azimuths.

    Returns:
    - Rotation matrices (batch_size x 3 x 3 tensor).
    - Translation vectors (batch_size x 3 tensor).
    """
    # Convert elevation and azimuth ranges from degrees to radians
    min_elevation_rad, max_elevation_rad = math.radians(
        elevation_range[0]
    ), math.radians(elevation_range[1])
    min_azimuth_rad, max_azimuth_rad = math.radians(azimuth_range[0]), math.radians(
        azimuth_range[1]
    )

    # Generate random distances, elevations, and azimuths for the batch
    distances = torch.FloatTensor(batch_size).uniform_(*distance_range)
    elevations = torch.FloatTensor(batch_size).uniform_(
        min_elevation_rad, max_elevation_rad
    )
    azimuths = torch.FloatTensor(batch_size).uniform_(min_azimuth_rad, max_azimuth_rad)

    # Convert spherical coordinates (distance, elevation, azimuth) to Cartesian coordinates (x, y, z)
    xs = distances * torch.cos(elevations) * torch.cos(azimuths)
    ys = distances * torch.cos(elevations) * torch.sin(azimuths)
    zs = distances * torch.sin(elevations)
    camera_positions = torch.stack([xs, ys, zs], dim=1)

    # Target position is the origin
    target_position = torch.zeros(1, 3)

    # Up direction is constant
    up_direction = torch.tensor([0.0, 1.0, 0.0]).expand(batch_size, -1)

    # Compute the forward vector for each camera
    forward = F.normalize(target_position - camera_positions, dim=1)

    # Compute the right vector for each camera
    right = F.normalize(torch.cross(up_direction, forward, dim=1), dim=1)

    # Compute the corrected up vector for each camera
    up = F.normalize(torch.cross(forward, right, dim=1), dim=1)

    # Construct the rotation matrices
    rotation_matrices = torch.stack([right, up, forward], dim=-1)

    # Translation vectors are just the negated camera positions
    translation_vectors = -camera_positions
    
    return rotation_matrices.to(device), translation_vectors.to(device)
