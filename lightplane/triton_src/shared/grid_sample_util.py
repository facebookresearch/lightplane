# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import triton
import triton.language as tl


@triton.jit
def _floor(x):
    return x - x % 1


@triton.jit
def _round(x):
    return _floor(x + 0.5)


@triton.jit
def is_in_bounds(
    x,
    y,
    z,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    in_bounds = (tl.abs(x) <= 1) * (tl.abs(y) <= 1) * (tl.abs(z) <= 1)
    if C == 1:
        in_bounds_mask = tl.view(in_bounds.to(tl.float32), (BLOCK_SIZE,))
    else:
        in_bounds_mask = tl.broadcast_to(
            in_bounds.to(tl.float32)[:, None], (BLOCK_SIZE, C)
        )
    return in_bounds_mask


@triton.jit
def _splat_3d(
    to_splat,
    grad_image,
    w,
    batch_index,
    ix,
    iy,
    iz,
    ID,
    IH,
    IW,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)
    iz_ = tl.minimum(tl.maximum(iz, 0.0), ID - 1).to(tl.int32)

    w = w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW) * (iz < ID) * (iz >= 0)).to(
        tl.float32
    )

    w = tl.view(w[:, None], (BLOCK_SIZE, 1))
    offs = tl.view(
        (batch_index * ID * IW * IH * C + iz_ * IW * IH * C + iy_ * IW * C + ix_ * C)[
            :, None
        ]
        + Coffs[None, :],
        (BLOCK_SIZE, C),
    )
    tl.atomic_add(grad_image + offs, w * to_splat)


@triton.jit
def _splat_2d(
    to_splat,
    grad_image,
    w,
    batch_index,
    ix,
    iy,
    IH,
    IW,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)

    w = w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW)).to(tl.float32)

    w = tl.view(w[:, None], (BLOCK_SIZE, 1))
    offs = tl.view(
        (batch_index * IW * IH * C + iy_ * IW * C + ix_ * C)[:, None] + Coffs[None, :],
        (BLOCK_SIZE, C),
    )
    tl.atomic_add(grad_image + offs, w * to_splat)


@triton.jit
def _voxel_grid_splat(
    to_splat,
    grad_feature_grid,
    feature_grid_size,
    batch_index,
    ix_in,
    iy_in,
    iz_in,
    C: tl.constexpr,
    NUM_GRIDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mask_out_of_bounds_samples: tl.constexpr,
):
    feature_grid_offs = tl.zeros((1,), dtype=tl.int32)
    for gi in range(NUM_GRIDS):
        offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

        ID = tl.load(feature_grid_size + offs + 1).to(tl.int32)
        IH = tl.load(feature_grid_size + offs + 2).to(tl.int32)
        IW = tl.load(feature_grid_size + offs + 3).to(tl.int32)

        ID_ = tl.sum(ID, axis=0) // BLOCK_SIZE
        IH_ = tl.sum(IH, axis=0) // BLOCK_SIZE
        IW_ = tl.sum(IW, axis=0) // BLOCK_SIZE
        voxel_grid = (ID_ - 1) * (IH_ - 1) * (IW_ - 1)

        if mask_out_of_bounds_samples:
            in_bounds_mask = is_in_bounds(
                ix_in,
                iy_in,
                iz_in,
                C,
                BLOCK_SIZE,
            )
            if C == 1:
                in_bounds_mask = in_bounds_mask[:, None]
            to_splat = to_splat * in_bounds_mask

        else:
            to_splat = to_splat

        if voxel_grid > 0:
            grid_numel = _voxel_grid_splat_one(
                gi,
                to_splat,
                grad_feature_grid + feature_grid_offs,
                feature_grid_size,
                batch_index,
                ix_in,
                iy_in,
                iz_in,
                IH,
                IW,
                ID,
                C,
                BLOCK_SIZE,
                mask_out_of_bounds_samples,
            )
        elif ID_ == 1:  # XY PLANE
            grid_numel = _plane_grid_splat_one(
                gi,
                to_splat,
                grad_feature_grid + feature_grid_offs,
                feature_grid_size,
                batch_index,
                ix_in,
                iy_in,
                IH,
                IW,
                C,
                BLOCK_SIZE,
                mask_out_of_bounds_samples,
            )
        elif IH_ == 1:  # XZ PLANE
            grid_numel = _plane_grid_splat_one(
                gi,
                to_splat,
                grad_feature_grid + feature_grid_offs,
                feature_grid_size,
                batch_index,
                ix_in,
                iz_in,
                ID,
                IW,
                C,
                BLOCK_SIZE,
                mask_out_of_bounds_samples,
            )
        else:  # YZ PLANE
            grid_numel = _plane_grid_splat_one(
                gi,
                to_splat,
                grad_feature_grid + feature_grid_offs,
                feature_grid_size,
                batch_index,
                iy_in,
                iz_in,
                ID,
                IH,
                C,
                BLOCK_SIZE,
                mask_out_of_bounds_samples,
            )
        feature_grid_offs += grid_numel


@triton.jit
def _get_voxel_grid_sample_info(
    gi,
    ix_in,
    iy_in,
    iz_in,
    ID,
    IH,
    IW,
    feature_grid_size,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    # load grid the sizes
    BS = tl.load(feature_grid_size + offs + 0).to(tl.int32)

    grid_numel = BS * ID * IH * IW * C  # size = (BLOCK_SIZE,)
    # just take one dim by averaging the result
    grid_numel = tl.sum(grid_numel, axis=0) // BLOCK_SIZE  # size = (1,)

    # map to [0, GRID_SIZE)
    ix11 = ((ix_in + 1) / 2) * IW.to(tl.float32) - 0.5
    iy11 = ((iy_in + 1) / 2) * IH.to(tl.float32) - 0.5
    iz11 = ((iz_in + 1) / 2) * ID.to(tl.float32) - 0.5

    # If a size along a certain dim is singleton, we set all its
    # coords to 0.0, this will make sure that the trilinear interp
    # will only index into the 0-th singleton dimension of the grid.
    ix = ix11 * (IW > 1).to(tl.float32)
    iy = iy11 * (IH > 1).to(tl.float32)
    iz = iz11 * (ID > 1).to(tl.float32)

    ix0 = _floor(ix).to(tl.float32)  # floor
    iy0 = _floor(iy).to(tl.float32)  # floor
    iz0 = _floor(iz).to(tl.float32)  # floor

    return ix, iy, iz, ix0, iy0, iz0, grid_numel


@triton.jit
def _get_plane_grid_sample_info(
    gi,
    ix_in,
    iy_in,
    IH,
    IW,
    feature_grid_size,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    # load grid the sizes
    BS = tl.load(feature_grid_size + offs + 0).to(tl.int32)

    grid_numel = BS * IH * IW * C  # size = (BLOCK_SIZE,)
    # just take one dim by averaging the result
    grid_numel = tl.sum(grid_numel, axis=0) // BLOCK_SIZE  # size = (1,)

    # map to [0, GRID_SIZE)
    ix11 = ((ix_in + 1) / 2) * IW.to(tl.float32) - 0.5
    iy11 = ((iy_in + 1) / 2) * IH.to(tl.float32) - 0.5

    # If a size along a certain dim is singleton, we set all its
    # coords to 0.0, this will make sure that the trilinear interp
    # will only index into the 0-th singleton dimension of the grid.
    ix = ix11 * (IW > 1).to(tl.float32)
    iy = iy11 * (IH > 1).to(tl.float32)

    ix0 = _floor(ix).to(tl.float32)  # floor
    iy0 = _floor(iy).to(tl.float32)  # floor

    return ix, iy, ix0, iy0, grid_numel


@triton.jit
def _get_voxel_grid_sample_locs_weights(ix, iy, iz, ix0, iy0, iz0):
    return (
        ix0,
        iy0,
        iz0,
        ix0,
        iy0,
        iz0 + 1,
        ix0,
        iy0 + 1,
        iz0,
        ix0 + 1,
        iy0,
        iz0,
        ix0 + 1,
        iy0,
        iz0 + 1,
        ix0 + 1,
        iy0 + 1,
        iz0,
        ix0,
        iy0 + 1,
        iz0 + 1,
        ix0 + 1,
        iy0 + 1,
        iz0 + 1,
        ix - ix0,
        iy - iy0,
        iz - iz0,
    )


@triton.jit
def _get_plane_grid_sample_locs_weights(ix, iy, ix0, iy0):
    return (
        ix0,
        iy0,
        ix0,
        iy0 + 1,
        ix0 + 1,
        iy0,
        ix0 + 1,
        iy0 + 1,
        ix - ix0,
        iy - iy0,
    )


@triton.jit
def _voxel_grid_splat_one(
    gi,
    to_splat,
    grad_feature_grid,
    feature_grid_size,
    batch_index,
    ix_in,
    iy_in,
    iz_in,
    IH,
    IW,
    ID,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mask_out_of_bounds_samples: tl.constexpr,
):
    ix, iy, iz, ix0, iy0, iz0, grid_numel = _get_voxel_grid_sample_info(
        gi,
        ix_in,
        iy_in,
        iz_in,
        ID,
        IH,
        IW,
        feature_grid_size,
        C,
        BLOCK_SIZE,
    )

    # V000 = data[ i   , j   ,  k   ].astype(np.int32)
    # V100 = data[(i+1), j   ,  k   ].astype(np.int32)
    # V010 = data[ i   ,(j+1),  k   ].astype(np.int32)
    # V001 = data[ i   , j   , (k+1)].astype(np.int32)
    # V101 = data[(i+1), j   , (k+1)].astype(np.int32)
    # V011 = data[ i   ,(j+1), (k+1)].astype(np.int32)
    # V110 = data[(i+1),(j+1),  k   ].astype(np.int32)
    # V111 = data[(i+1),(j+1), (k+1)].astype(np.int32)

    (
        V000x,
        V000y,
        V000z,
        V100x,
        V100y,
        V100z,
        V010x,
        V010y,
        V010z,
        V001x,
        V001y,
        V001z,
        V101x,
        V101y,
        V101z,
        V011x,
        V011y,
        V011z,
        V110x,
        V110y,
        V110z,
        V111x,
        V111y,
        V111z,
        x,
        y,
        z,
    ) = _get_voxel_grid_sample_locs_weights(ix, iy, iz, ix0, iy0, iz0)

    _splat_3d(
        to_splat,
        grad_feature_grid,
        (1 - x) * (1 - y) * (1 - z),
        batch_index,
        V000x,
        V000y,
        V000z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_feature_grid,
        (1 - x) * (1 - y) * z,
        batch_index,
        V100x,
        V100y,
        V100z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_feature_grid,
        (1 - x) * y * (1 - z),
        batch_index,
        V010x,
        V010y,
        V010z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_feature_grid,
        x * (1 - y) * (1 - z),
        batch_index,
        V001x,
        V001y,
        V001z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_feature_grid,
        x * (1 - y) * z,
        batch_index,
        V101x,
        V101y,
        V101z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_feature_grid,
        x * y * (1 - z),
        batch_index,
        V011x,
        V011y,
        V011z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_feature_grid,
        (1 - x) * y * z,
        batch_index,
        V110x,
        V110y,
        V110z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    _splat_3d(
        to_splat,
        grad_feature_grid,
        x * y * z,
        batch_index,
        V111x,
        V111y,
        V111z,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )

    return grid_numel


@triton.jit
def _plane_grid_splat_one(
    gi,
    to_splat,
    grad_feature_grid,
    feature_grid_size,
    batch_index,
    ix_in,
    iy_in,
    IH,
    IW,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mask_out_of_bounds_samples: tl.constexpr,
):
    ix, iy, ix0, iy0, grid_numel = _get_plane_grid_sample_info(
        gi,
        ix_in,
        iy_in,
        IH,
        IW,
        feature_grid_size,
        C,
        BLOCK_SIZE,
    )

    # V00 = data[ i   , j ].astype(np.int32)
    # V10 = data[(i+1), j ].astype(np.int32)
    # V01 = data[ i   ,(j+1)].astype(np.int32)
    # V11 = data[(i+1),(j+1)].astype(np.int32)

    (
        V00x,
        V00y,
        V10x,
        V10y,
        V01x,
        V01y,
        V11x,
        V11y,
        x,
        y,
    ) = _get_plane_grid_sample_locs_weights(ix, iy, ix0, iy0)

    # if mask_out_of_bounds_samples:
    #     in_bounds_mask = is_in_bounds_2D(
    #         ix_in,
    #         iy_in,
    #         C,
    #         BLOCK_SIZE,
    #     )
    #     to_splat_now = to_splat * in_bounds_mask
    # else:
    #     to_splat_now = to_splat

    # Vxy = (V00 * (1 - x)*(1 - y)
    #         + V10 * x * (1 - y) * (1 - z) +
    #         + V010 * (1 - x) * y * (1 - z) +
    #         + V001 * (1 - x) * (1 - y) * z +
    #         + V101 * x * (1 - y) * z +
    #         + V011 * (1 - x) * y * z +
    #         + V110 * x * y * (1 - z) +
    #         + V111 * x * y * z)
    to_splat_now = to_splat
    _splat_2d(
        to_splat_now,
        grad_feature_grid,
        (1 - x) * (1 - y),
        batch_index,
        V00x,
        V00y,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )

    _splat_2d(
        to_splat_now,
        grad_feature_grid,
        (1 - x) * y,
        batch_index,
        V10x,
        V10y,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )

    _splat_2d(
        to_splat_now,
        grad_feature_grid,
        x * (1 - y),
        batch_index,
        V01x,
        V01y,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )

    _splat_2d(
        to_splat_now,
        grad_feature_grid,
        x * y,
        batch_index,
        V11x,
        V11y,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )
    return grid_numel


@triton.jit
def _sample_3d(
    image,
    w,
    batch_index,
    ix,
    iy,
    iz,
    ID,
    IH,
    IW,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)
    iz_ = tl.minimum(tl.maximum(iz, 0.0), ID - 1).to(tl.int32)

    image_offs = (
        image
        + batch_index * ID * IW * IH * C
        + iz_ * IW * IH * C
        + iy_ * IW * C
        + ix_ * C
    )

    mask_w = w * (
        (iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW) * (iz < ID) * (iz >= 0)
    ).to(tl.float32)

    if C == 1:  # do not append the last dim
        val = tl.view(tl.load(image_offs).to(tl.float32), (BLOCK_SIZE,))
        out = tl.view(val * mask_w, (BLOCK_SIZE,))
        return out

    else:
        val = tl.view(
            tl.load(image_offs[:, None] + Coffs[None, :]).to(tl.float32),
            (BLOCK_SIZE, C),
        )
        mask_w_bcast = tl.view(mask_w[:, None], (BLOCK_SIZE, 1))
        return val * mask_w_bcast


@triton.jit
def _sample_2d(
    image,
    w,
    batch_index,
    ix,
    iy,
    IH,
    IW,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1).to(tl.int32)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1).to(tl.int32)

    image_offs = image + batch_index * IW * IH * C + iy_ * IW * C + ix_ * C

    mask_w = w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW)).to(tl.float32)

    if C == 1:  # do not append the last dim
        val = tl.view(tl.load(image_offs).to(tl.float32), (BLOCK_SIZE,))
        out = tl.view(val * mask_w, (BLOCK_SIZE,))
        return out

    else:
        val = tl.view(
            tl.load(image_offs[:, None] + Coffs[None, :]).to(tl.float32),
            (BLOCK_SIZE, C),
        )
        mask_w_bcast = tl.view(mask_w[:, None], (BLOCK_SIZE, 1))
        return val * mask_w_bcast


@triton.jit
def voxel_grid_sample_one_nearest(
    gi,
    feature_grid,
    feature_grid_size,
    batch_index,
    ix_in,
    iy_in,
    iz_in,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mask_out_of_bounds_samples: tl.constexpr,
):
    offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    ID = tl.load(feature_grid_size + offs + 1).to(tl.int32)
    IH = tl.load(feature_grid_size + offs + 2).to(tl.int32)
    IW = tl.load(feature_grid_size + offs + 3).to(tl.int32)

    # map to [0, GRID_SIZE)
    ix11 = ((ix_in + 1) / 2) * IW - 0.5
    iy11 = ((iy_in + 1) / 2) * IH - 0.5
    iz11 = ((iz_in + 1) / 2) * ID - 0.5

    # If a size along a certain dim is singleton, we set all its
    # coords to 0.0, this will make sure that the trilinear interp
    # will only index into the 0-th singleton dimension of the grid.
    ix = ix11 * (ID > 1).to(tl.float32)
    iy = iy11 * (IH > 1).to(tl.float32)
    iz = iz11 * (IW > 1).to(tl.float32)
    unit_weight = ix * 0.0 + 1.0

    ix = _round(ix).to(tl.int32)
    iy = _round(iy).to(tl.int32)
    iz = _round(iz).to(tl.int32)

    sampled = _sample_3d(
        feature_grid,
        unit_weight,
        batch_index,
        ix,
        iy,
        iz,
        ID,
        IH,
        IW,
        C,
        BLOCK_SIZE,
    )

    if mask_out_of_bounds_samples:
        in_bounds_mask = is_in_bounds(
            ix_in,
            iy_in,
            iz_in,
            C,
            BLOCK_SIZE,
        )
        sampled *= in_bounds_mask

    return sampled


@triton.jit
def _voxel_grid_sample_one(
    gi,
    feature_grid,
    feature_grid_size,
    batch_index,
    ix_in,
    iy_in,
    iz_in,
    ID,
    IH,
    IW,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mask_out_of_bounds_samples: tl.constexpr,
):
    ix, iy, iz, ix0, iy0, iz0, grid_numel = _get_voxel_grid_sample_info(
        gi,
        ix_in,
        iy_in,
        iz_in,
        ID,
        IH,
        IW,
        feature_grid_size,
        C,
        BLOCK_SIZE,
    )

    # V000 = data[ i   , j   ,  k   ].astype(np.int32)
    # V100 = data[(i+1), j   ,  k   ].astype(np.int32)
    # V010 = data[ i   ,(j+1),  k   ].astype(np.int32)
    # V001 = data[ i   , j   , (k+1)].astype(np.int32)
    # V101 = data[(i+1), j   , (k+1)].astype(np.int32)
    # V011 = data[ i   ,(j+1), (k+1)].astype(np.int32)
    # V110 = data[(i+1),(j+1),  k   ].astype(np.int32)
    # V111 = data[(i+1),(j+1), (k+1)].astype(np.int32)

    (
        V000x,
        V000y,
        V000z,
        V100x,
        V100y,
        V100z,
        V010x,
        V010y,
        V010z,
        V001x,
        V001y,
        V001z,
        V101x,
        V101y,
        V101z,
        V011x,
        V011y,
        V011z,
        V110x,
        V110y,
        V110z,
        V111x,
        V111y,
        V111z,
        x,
        y,
        z,
    ) = _get_voxel_grid_sample_locs_weights(ix, iy, iz, ix0, iy0, iz0)

    # Vxyz = (V000 * (1 - x)*(1 - y)*(1 - z)
    #         + V100 * x * (1 - y) * (1 - z) +
    #         + V010 * (1 - x) * y * (1 - z) +
    #         + V001 * (1 - x) * (1 - y) * z +
    #         + V101 * x * (1 - y) * z +
    #         + V011 * (1 - x) * y * z +
    #         + V110 * x * y * (1 - z) +
    #         + V111 * x * y * z)

    sampled = (
        _sample_3d(
            feature_grid,
            (1 - x) * (1 - y) * (1 - z),
            batch_index,
            V000x,
            V000y,
            V000z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            feature_grid,
            (1 - x) * (1 - y) * z,
            batch_index,
            V100x,
            V100y,
            V100z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            feature_grid,
            (1 - x) * y * (1 - z),
            batch_index,
            V010x,
            V010y,
            V010z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            feature_grid,
            x * (1 - y) * (1 - z),
            batch_index,
            V001x,
            V001y,
            V001z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            feature_grid,
            x * (1 - y) * z,
            batch_index,
            V101x,
            V101y,
            V101z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            feature_grid,
            x * y * (1 - z),
            batch_index,
            V011x,
            V011y,
            V011z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            feature_grid,
            (1 - x) * y * z,
            batch_index,
            V110x,
            V110y,
            V110z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_3d(
            feature_grid,
            x * y * z,
            batch_index,
            V111x,
            V111y,
            V111z,
            ID,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
    )

    # if mask_out_of_bounds_samples:
    #     in_bounds_mask = is_in_bounds(
    #         ix_in,
    #         iy_in,
    #         iz_in,
    #         C,
    #         BLOCK_SIZE,
    #     )
    #     sampled *= in_bounds_mask

    return sampled, grid_numel


@triton.jit
def _plane_grid_sample_one(
    gi,
    feature_grid,
    feature_grid_size,
    batch_index,
    ix_in,
    iy_in,
    IH,
    IW,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mask_out_of_bounds_samples: tl.constexpr,
):
    ix, iy, ix0, iy0, grid_numel = _get_plane_grid_sample_info(
        gi,
        ix_in,
        iy_in,
        IH,
        IW,
        feature_grid_size,
        C,
        BLOCK_SIZE,
    )

    # V00 = data[ i   , j ].astype(np.int32)
    # V10 = data[(i+1), j ].astype(np.int32)
    # V01 = data[ i   ,(j+1)].astype(np.int32)
    # V11 = data[(i+1),(j+1)].astype(np.int32)

    (
        V00x,
        V00y,
        V10x,
        V10y,
        V01x,
        V01y,
        V11x,
        V11y,
        x,
        y,
    ) = _get_plane_grid_sample_locs_weights(ix, iy, ix0, iy0)

    # Vxy = (V00 * (1 - x)*(1 - y)
    #         + V10 * x * (1 - y) * (1 - z) +
    #         + V010 * (1 - x) * y * (1 - z) +
    #         + V001 * (1 - x) * (1 - y) * z +
    #         + V101 * x * (1 - y) * z +
    #         + V011 * (1 - x) * y * z +
    #         + V110 * x * y * (1 - z) +
    #         + V111 * x * y * z)

    sampled = (
        _sample_2d(
            feature_grid,
            (1 - x) * (1 - y),
            batch_index,
            V00x,
            V00y,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_2d(
            feature_grid,
            x * (1 - y),
            batch_index,
            V01x,
            V01y,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_2d(
            feature_grid,
            (1 - x) * y,
            batch_index,
            V10x,
            V10y,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
        + _sample_2d(
            feature_grid,
            x * y,
            batch_index,
            V11x,
            V11y,
            IH,
            IW,
            C,
            BLOCK_SIZE,
        )
    )

    # if mask_out_of_bounds_samples:
    #     in_bounds_mask = is_in_bounds_2D(
    #         ix_in,
    #         iy_in,
    #         C,
    #         BLOCK_SIZE,
    #     )
    #     sampled *= in_bounds_mask

    return sampled, grid_numel


@triton.jit
def _voxel_grid_sample(
    feature_grid,
    feature_grid_size,
    batch_index,
    ix_in,
    iy_in,
    iz_in,
    C: tl.constexpr,
    NUM_GRIDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mask_out_of_bounds_samples: tl.constexpr,
):
    out_val = tl.zeros((BLOCK_SIZE, C), dtype=tl.float32)
    feature_grid_offs = tl.zeros((1,), dtype=tl.int32)

    for gi in range(NUM_GRIDS):
        offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

        ID = tl.load(feature_grid_size + offs + 1).to(tl.int32)
        IH = tl.load(feature_grid_size + offs + 2).to(tl.int32)
        IW = tl.load(feature_grid_size + offs + 3).to(tl.int32)

        ID_ = tl.sum(ID, axis=0) // BLOCK_SIZE
        IH_ = tl.sum(IH, axis=0) // BLOCK_SIZE
        IW_ = tl.sum(IW, axis=0) // BLOCK_SIZE
        voxel_grid = (ID_ - 1) * (IH_ - 1) * (IW_ - 1)

        if voxel_grid > 0:
            sampled, grid_numel = _voxel_grid_sample_one(
                gi,
                feature_grid + feature_grid_offs,
                feature_grid_size,
                batch_index,
                ix_in,
                iy_in,
                iz_in,
                ID,
                IH,
                IW,
                C,
                BLOCK_SIZE,
                mask_out_of_bounds_samples,
            )
        elif ID_ == 1:  # XY PLANE
            sampled, grid_numel = _plane_grid_sample_one(
                gi,
                feature_grid + feature_grid_offs,
                feature_grid_size,
                batch_index,
                ix_in,
                iy_in,
                IH,
                IW,
                C,
                BLOCK_SIZE,
                mask_out_of_bounds_samples,
            )
        elif IH_ == 1:  # XZ PLANE
            sampled, grid_numel = _plane_grid_sample_one(
                gi,
                feature_grid + feature_grid_offs,
                feature_grid_size,
                batch_index,
                ix_in,
                iz_in,
                ID,
                IW,
                C,
                BLOCK_SIZE,
                mask_out_of_bounds_samples,
            )
        else:  # YZ PLANE
            sampled, grid_numel = _plane_grid_sample_one(
                gi,
                feature_grid + feature_grid_offs,
                feature_grid_size,
                batch_index,
                iy_in,
                iz_in,
                ID,
                IH,
                C,
                BLOCK_SIZE,
                mask_out_of_bounds_samples,
            )
        out_val += sampled
        feature_grid_offs += grid_numel
    # we mask results together.
    if mask_out_of_bounds_samples:
        in_bounds_mask = is_in_bounds(
            ix_in,
            iy_in,
            iz_in,
            C,
            BLOCK_SIZE,
        )
        out_val *= in_bounds_mask
    return out_val


@triton.jit
def sample_grid_rep(
    feature_grid,
    feature_grid_sizes,
    grid_idx,
    sample_x,
    sample_y,
    sample_z,
    C: tl.constexpr,
    NUM_GRIDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mask_out_of_bounds_samples: tl.constexpr,
):
    # we use only voxel grid sample, triplane sampling is a special case of that
    vec = _voxel_grid_sample(
        feature_grid,
        feature_grid_sizes,
        grid_idx,
        sample_x,
        sample_y,
        sample_z,
        C,
        NUM_GRIDS,
        BLOCK_SIZE,
        mask_out_of_bounds_samples,
    )

    return vec


@triton.jit
def splat_grid_rep(
    feature_grid,
    grad_image,
    feature_grid_sizes,
    grid_idx,
    sample_x,
    sample_y,
    sample_z,
    C: tl.constexpr,
    NUM_GRIDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mask_out_of_bounds_samples: tl.constexpr,
):
    # we use only voxel grid splat, triplane splatting is a special case of that
    _voxel_grid_splat(
        feature_grid,
        grad_image,
        feature_grid_sizes,
        grid_idx,
        sample_x,
        sample_y,
        sample_z,
        C,
        NUM_GRIDS,
        BLOCK_SIZE,
        mask_out_of_bounds_samples,
    )
