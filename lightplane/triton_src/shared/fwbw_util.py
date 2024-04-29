# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import triton
import triton.language as tl


@triton.jit
def fwbw_init(
    directions,
    origins,
    grid_idx,
    near,
    far,
    rays_encoding,
    inject_noise_seed,
    # ---- mlp params ----
    DIM_IN_COLOR: tl.constexpr,
    DIM_OUT_COLOR: tl.constexpr,
    # ----- config keys ----
    num_samples: tl.constexpr,
    num_samples_inf: tl.constexpr,
    # ----- sizes ----
    num_rays: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tot_num_samples = num_samples + num_samples_inf
    pid = tl.program_id(axis=0)

    # the number of output dims is the last entry in mlp_n_hidden:
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_mask = offs < num_rays

    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1

    offs_features = (
        pid * BLOCK_SIZE * DIM_OUT_COLOR
        + DIM_OUT_COLOR * tl.arange(0, BLOCK_SIZE)[:, None]
        + tl.arange(0, DIM_OUT_COLOR)[None, :]
    )
    offs_features_mask = (
        pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    ) < num_rays

    center_x = tl.load(origins + offs_x, mask=offs_x < num_rays * 3).to(tl.float32)
    center_y = tl.load(origins + offs_y, mask=offs_y < num_rays * 3).to(tl.float32)
    center_z = tl.load(origins + offs_z, mask=offs_z < num_rays * 3).to(tl.float32)

    ray_x = tl.load(directions + offs_x, mask=offs_x < num_rays * 3).to(tl.float32)
    ray_y = tl.load(directions + offs_y, mask=offs_y < num_rays * 3).to(tl.float32)
    ray_z = tl.load(directions + offs_z, mask=offs_z < num_rays * 3).to(tl.float32)

    # batch_index = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) // num_rays_per_batch
    near_buffer = tl.load(near + offs, mask=offs_mask).to(tl.float32)
    far_buffer = tl.load(far + offs, mask=offs_mask).to(tl.float32)
    grid_idx_buffer = tl.load(grid_idx + offs, mask=offs_mask).to(tl.int32)

    seed_buffer = tl.load(inject_noise_seed + offs, mask=offs < num_rays).to(tl.int32)
    sample_index_buffer = (
        tl.arange(0, BLOCK_SIZE) * tot_num_samples
        + pid * BLOCK_SIZE * tot_num_samples
        + 1
    )

    rays_encoding_buffer = tl.load(
        rays_encoding
        + pid * BLOCK_SIZE * DIM_IN_COLOR
        + DIM_IN_COLOR * tl.arange(0, BLOCK_SIZE)[:, None]
        + tl.arange(0, DIM_IN_COLOR)[None, :],
        mask=offs_features_mask,
    )

    one_scaffold = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
    zero_value = tl.zeros((BLOCK_SIZE,), tl.float32)
    one_vec = tl.full((BLOCK_SIZE, C), 1.0, tl.float32)
    zero_color = tl.zeros((BLOCK_SIZE, DIM_OUT_COLOR), tl.float32)

    return (
        tot_num_samples,
        pid,
        offs,
        offs_mask,
        offs_features,
        offs_features_mask,
        center_x,
        center_y,
        center_z,
        ray_x,
        ray_y,
        ray_z,
        near_buffer,
        far_buffer,
        grid_idx_buffer,
        seed_buffer,
        sample_index_buffer,
        rays_encoding_buffer,
        one_scaffold,
        zero_value,
        one_vec,
        zero_color,
    )


@triton.jit
def fwbw_splatter_init(
    directions,
    origins,
    grid_idx,
    near,
    far,
    splatting_feature,
    mask,
    # ----- config keys ----
    num_samples: tl.constexpr,
    num_samples_inf: tl.constexpr,
    # ----- sizes ----
    num_rays: tl.constexpr,
    grid_channel: tl.constexpr,
    feature_channel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tot_num_samples = num_samples + num_samples_inf
    pid = tl.program_id(axis=0)

    # the number of output dims is the last entry in mlp_n_hidden:
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_mask = offs < num_rays

    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1

    offs_features = (
        pid * BLOCK_SIZE * feature_channel
        + feature_channel * tl.arange(0, BLOCK_SIZE)[:, None]
        + tl.arange(0, feature_channel)[None, :]
    )
    offs_features_mask = (
        pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    ) < num_rays

    center_x = tl.load(origins + offs_x, mask=offs_x < num_rays * 3).to(tl.float32)
    center_y = tl.load(origins + offs_y, mask=offs_y < num_rays * 3).to(tl.float32)
    center_z = tl.load(origins + offs_z, mask=offs_z < num_rays * 3).to(tl.float32)

    ray_x = tl.load(directions + offs_x, mask=offs_x < num_rays * 3).to(tl.float32)
    ray_y = tl.load(directions + offs_y, mask=offs_y < num_rays * 3).to(tl.float32)
    ray_z = tl.load(directions + offs_z, mask=offs_z < num_rays * 3).to(tl.float32)

    # batch_index = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) // num_rays_per_batch
    near_buffer = tl.load(near + offs, mask=offs_mask).to(tl.float32)
    far_buffer = tl.load(far + offs, mask=offs_mask).to(tl.float32)
    grid_idx_buffer = tl.load(grid_idx + offs, mask=offs_mask).to(tl.int32)
    sample_index_buffer = (
        tl.arange(0, BLOCK_SIZE) * tot_num_samples
        + pid * BLOCK_SIZE * tot_num_samples
        + 1
    )

    feature = tl.load(splatting_feature + offs_features, mask=offs_features_mask).to(
        tl.float32
    )

    mask = tl.load(
        mask + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None],
        mask=offs_features_mask,
    ).to(tl.float32)
    mask = tl.view(mask, (BLOCK_SIZE, 1))
    return (
        tot_num_samples,
        pid,
        offs,
        offs_mask,
        offs_features,
        offs_features_mask,
        center_x,
        center_y,
        center_z,
        ray_x,
        ray_y,
        ray_z,
        near_buffer,
        far_buffer,
        grid_idx_buffer,
        sample_index_buffer,
        feature,
        mask,
    )
