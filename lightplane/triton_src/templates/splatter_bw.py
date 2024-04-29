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
# x_str_mlp = cog_util.get_x_str("mlp", N_LAYERS)
# xwb_str_mlp = cog_util.get_xwb_str("mlp", N_LAYERS)
# dwb_str_mlp = cog_util.get_dwb_str("mlp", N_LAYERS)
#
# mlp_util_file = cog_util.get_generated_splatter_file_name(
#   "splatter_mlp_util",
#   N_LAYERS
# )
# cog.outl(f"from lightplane.triton_src.generated.{mlp_util_file} import load_mlp_params, mlp_splatter, mlp_splatter_with_inter_feat, update_mlp_params, d_mlp_splatter, init_mlp_params_grads")
# ]]]
# [[[end]]]


@triton.jit
def bw_kernel(
    grad_feature_grid,
    grad_feature_grid_sizes,
    # ---- grid ----
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
    # ---- outputs ----
    grad_splatting_feature,
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

    depth = near_buffer
    grad_splatting_feature_buffer = tl.zeros(
        (BLOCK_SIZE, feature_channel), dtype=tl.float32
    )

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

        grad_vec = sample_grid_rep(
            grad_feature_grid,
            grad_feature_grid_sizes,
            grid_idx_buffer,
            sample_x,
            sample_y,
            sample_z,
            grid_channel,
            NUM_GRIDS,
            BLOCK_SIZE,
            mask_out_of_bounds_samples,
        )
        grad_vec = grad_vec * mask_buffer
        grad_splatting_feature_buffer += grad_vec
    tl.store(
        grad_splatting_feature + offs_features,
        grad_splatting_feature_buffer,
        mask=offs_features_mask,
    )


@triton.jit
def bw_kernel_wMLP(
    grad_feature_grid,
    grad_feature_grid_sizes,
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
    # ---- outputs ----
    grad_splatting_feature,
    grad_mlp_params,
    grad_input_feature_grid,
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

    # [[[cog
    # if N_LAYERS > 0:
    #   cog.outl("(")
    #   value_str = cog_util.get_dwb_str("acc", N_LAYERS)
    #   cog.outl(f" {value_str},")
    #   cog.outl(") = init_mlp_params_grads(")
    # else:
    #   cog.outl("init_mlp_params_grads(")
    # cog.outl("    DIM_HIDDEN,")
    # cog.outl("    DIM_IN,")
    # cog.outl("    DIM_OUT")
    # cog.outl(")")
    # ]]]
    # [[[end]]]
    depth = near_buffer
    grad_splatting_feature_buffer = tl.zeros(
        (BLOCK_SIZE, feature_channel), dtype=tl.float32
    )

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

        grad_vec = sample_grid_rep(
            grad_feature_grid,
            grad_feature_grid_sizes,
            grid_idx_buffer,
            sample_x,
            sample_y,
            sample_z,
            grid_channel,
            NUM_GRIDS,
            BLOCK_SIZE,
            mask_out_of_bounds_samples,
        )
        grad_vec = grad_vec * mask_buffer
        fused_feature = feature_buffer + prev_vec
        # [[[cog
        # def create_grad_accum_function( n_layers):
        #   for l in range(n_layers):
        #       dim_out = f"DIM_OUT" if (l == int(n_layers) - 1) else f"DIM_HIDDEN"
        #       cog.outl(f"dw{l}_acc += dw{l}_mlp")
        #       cog.outl(f"db{l}_acc += db{l}_mlp")
        #
        # if N_LAYERS > 0:
        #   cog.outl("(fused_feature, " + x_str_mlp + ", " + xwb_str_mlp + ") =  mlp_splatter_with_inter_feat(")
        #   cog.outl("    fused_feature,")
        #   cog.outl("    " + wb_str_mlp + ",")
        #   cog.outl(")")
        #   cog.outl(" ")
        #   cog.outl(f"grad_splatting, "+ dwb_str_mlp + " = d_mlp_splatter(")
        #   cog.outl(f"    grad_vec,")
        #   cog.outl(f"    " + wb_str_mlp + ", " + xwb_str_mlp + ", " + x_str_mlp + ")")
        #   cog.outl(" ")
        #   create_grad_accum_function(N_LAYERS)
        # else:
        #   cog.outl("fused_feature = mlp_splatter_with_inter_feat(fused_feature)")
        #   cog.outl("grad_splatting = d_mlp_splatter(fused_feature)")
        # ]]]
        # [[[end]]]

        splat_grid_rep(
            grad_splatting,
            grad_input_feature_grid,
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
        grad_splatting_feature_buffer += grad_splatting
    tl.store(
        grad_splatting_feature + offs_features,
        grad_splatting_feature_buffer,
        mask=offs_features_mask,
    )
    update_mlp_params(
        grad_mlp_params,
        DIM_IN,
        DIM_HIDDEN,
        # [[[cog
        # if N_LAYERS > 0:
        #   cog.outl("DIM_OUT,")
        #   mlp_str = cog_util.get_dwb_str("acc", N_LAYERS)
        #   cog.outl(f"{mlp_str}")
        # else:
        #   cog.outl("DIM_OUT")
        # ]]]
        # [[[end]]]
    )
