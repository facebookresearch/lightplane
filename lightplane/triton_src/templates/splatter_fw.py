# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# example call: > cog -d -D N_LAYERS=2 COG_UTIL_MODULE=lightplane.triton_src.templates.cog_util MLP_UTIL_MODULE=lightplane.triton_src.generated.mlp_util_2 ./mlp_util.py
# This `cog` template expects the following constants:
#  N_LAYERS: number of mlp layers
#  COG_UTIL_MODULE: the module name of the cog_util.py file
#  MLP_UTIL_MODULE: the module name of the mlp_util.py file

import triton
import triton.language as tl

from lightplane.triton_src.shared.func_util import softplus
from lightplane.triton_src.shared.fwbw_util import fwbw_splatter_init
from lightplane.triton_src.shared.grid_sample_util import (
    sample_grid_rep,
    splat_grid_rep,
)
from lightplane.triton_src.shared.rand_util import int_to_randn
from lightplane.triton_src.shared.ray_util import (
    contract_pi,
    depth_inv_sphere,
    depth_lin,
)


# these functions will be overridden by imports from the auto-generated cog code below:
def load_mlp_params():
    pass


def mlp_trunk():
    pass


def mlp_color():
    pass


def mlp_opacity():
    pass


# auto-generated cog import code
# [[[cog
# import cog
# import os
# import importlib
# import inspect
# import sys
# N_LAYERS = int(N_LAYERS)
# cog_util = importlib.import_module(COG_UTIL_MODULE)
# mlp_util = importlib.import_module(MLP_UTIL_MODULE)
#
# # these strings are needed later:
# wb_str_mlp = cog_util.get_wb_str("mlp", N_LAYERS)
#
# mlp_util_file = cog_util.get_generated_splatter_file_name(
#   "splatter_mlp_util",
#   N_LAYERS
# )
# cog.outl(f"from lightplane.triton_src.generated.{mlp_util_file} import load_mlp_params, mlp_splatter")
# ]]]
# [[[end]]]


@triton.jit
def fw_kernel(
    # ---- grid ----
    feature_grid,
    feature_grid_sizes,
    # ----- non-differentiable tensors
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
    NUM_GRIDS: tl.constexpr,
    feature_channel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    # ---- switches ----
    mask_out_of_bounds_samples: tl.constexpr,
    contract_coords: tl.constexpr,
    disparity_at_inf: tl.constexpr,
):
    (
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
        feature_buffer,
        mask_buffer,
    ) = fwbw_splatter_init(
        directions,
        origins,
        grid_idx,
        near,
        far,
        splatting_feature,
        mask,
        num_samples,
        num_samples_inf,
        num_rays,
        grid_channel,
        feature_channel,
        BLOCK_SIZE,
    )

    feature_buffer = feature_buffer * mask_buffer

    for step in range(tot_num_samples):
        if step < num_samples:
            depth = depth_lin(near_buffer, far_buffer, num_samples, step)
        else:
            depth = depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                step - num_samples,
            )

        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z

        if contract_coords:
            sample_x, sample_y, sample_z = contract_pi(sample_x, sample_y, sample_z)

        splat_grid_rep(
            feature_buffer,
            feature_grid,
            feature_grid_sizes,
            grid_idx_buffer,
            sample_x,
            sample_y,
            sample_z,
            grid_channel,
            NUM_GRIDS,
            BLOCK_SIZE,
            mask_out_of_bounds_samples,
        )


@triton.jit
def fw_kernel_wMLP(
    # ---- grid ----
    feature_grid,
    feature_grid_sizes,
    input_feature_grid,
    input_feature_grid_sizes,
    # ----- non-differentiable tensors
    directions,
    origins,
    grid_idx,
    near,
    far,
    splatting_feature,
    mask,
    # ---- mlp params ----
    mlp_params,  # master ptr for the mlp params
    DIM_HIDDEN: tl.constexpr,
    DIM_IN: tl.constexpr,
    DIM_OUT: tl.constexpr,
    # ----- config keys ----
    num_samples: tl.constexpr,
    num_samples_inf: tl.constexpr,
    # ----- sizes ----
    num_rays: tl.constexpr,
    grid_channel: tl.constexpr,
    NUM_GRIDS: tl.constexpr,
    feature_channel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    # ---- switches ----
    mask_out_of_bounds_samples: tl.constexpr,
    contract_coords: tl.constexpr,
    disparity_at_inf: tl.constexpr,
):
    (
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
        feature_buffer,
        mask_buffer,
    ) = fwbw_splatter_init(
        directions,
        origins,
        grid_idx,
        near,
        far,
        splatting_feature,
        mask,
        num_samples,
        num_samples_inf,
        num_rays,
        grid_channel,
        feature_channel,
        BLOCK_SIZE,
    )
    # [[[cog
    # if N_LAYERS > 0:
    #   cog.outl("(")
    #   cog.outl("  " + wb_str_mlp + ("," if len(wb_str_mlp) > 0 else ""))
    #   cog.outl(") = load_mlp_params(")
    # else:
    #   cog.outl("load_mlp_params(")
    # cog.outl("    mlp_params,  # master ptr for the mlp params")
    # cog.outl("    DIM_IN,")
    # cog.outl("    DIM_HIDDEN,")
    # cog.outl("    DIM_OUT,")
    # cog.outl("    BLOCK_SIZE")
    # cog.outl(")")
    # ]]]
    # [[[end]]]

    for step in range(tot_num_samples):
        if step < num_samples:
            depth = depth_lin(near_buffer, far_buffer, num_samples, step)
        else:
            depth = depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                step - num_samples,
            )

        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z

        if contract_coords:
            sample_x, sample_y, sample_z = contract_pi(sample_x, sample_y, sample_z)

        prev_vec = sample_grid_rep(
            input_feature_grid,
            input_feature_grid_sizes,
            grid_idx_buffer,
            sample_x,
            sample_y,
            sample_z,
            feature_channel,
            NUM_GRIDS,
            BLOCK_SIZE,
            mask_out_of_bounds_samples,
        )

        fused_feature = feature_buffer + prev_vec

        # [[[cog
        # if N_LAYERS > 0:
        #   cog.outl("fused_feature = mlp_splatter(")
        #   cog.outl("    fused_feature,")
        #   cog.outl("    " + wb_str_mlp + " ")
        #   cog.outl(")")
        # else:
        #   cog.outl("fused_feature = mlp_splatter(fused_feature)")
        # ]]]
        # [[[end]]]

        fused_feature = fused_feature * mask_buffer
        splat_grid_rep(
            fused_feature,
            feature_grid,
            feature_grid_sizes,
            grid_idx_buffer,
            sample_x,
            sample_y,
            sample_z,
            grid_channel,
            NUM_GRIDS,
            BLOCK_SIZE,
            mask_out_of_bounds_samples,
        )
