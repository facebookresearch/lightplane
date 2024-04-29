# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# example call: > cog -d -D N_LAYERS_TRUNK=2 -D N_LAYERS_OPACITY=1 -D N_LAYERS_COLOR=1 COG_UTIL_MODULE=lightplane.triton_src.templates.cog_util MLP_UTIL_MODULE=lightplane.triton_src.generated.mlp_util_t2_o1_c2 ./mlp_util.py
# This `cog` template expects the following constants:
#  N_LAYERS_TRUNK: number of mlp layers in trunk
#  N_LAYERS_OPACITY: number of mlp layers in opacity head
#  N_LAYERS_COLOR: number of mlp layers in color head
#  COG_UTIL_MODULE: the module name of the cog_util.py file
#  MLP_UTIL_MODULE: the module name of the mlp_util.py file


import triton
import triton.language as tl

from lightplane.triton_src.shared.func_util import softplus
from lightplane.triton_src.shared.fwbw_util import fwbw_init
from lightplane.triton_src.shared.grid_sample_util import (
    sample_grid_rep,
    voxel_grid_sample_one_nearest,
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


#fmt: off
#auto-generated cog import code
#[[[cog
#import cog
#import os
#import importlib
#import inspect
#import sys
#N_LAYERS_TRUNK, N_LAYERS_OPACITY, N_LAYERS_COLOR = int(N_LAYERS_TRUNK), int(N_LAYERS_OPACITY), int(N_LAYERS_COLOR)
#cog_util = importlib.import_module(COG_UTIL_MODULE)
#mlp_util = importlib.import_module(MLP_UTIL_MODULE)
#
#N_LAYERS_TRUNK, N_LAYERS_OPACITY, N_LAYERS_COLOR = int(N_LAYERS_TRUNK), int(N_LAYERS_OPACITY), int(N_LAYERS_COLOR)
#
##these strings are needed later:
#wb_str_trunk = cog_util.get_wb_str("trunk", N_LAYERS_TRUNK)
#wb_str_opacity = cog_util.get_wb_str("opacity", N_LAYERS_OPACITY)
#wb_str_color = cog_util.get_wb_str("color", N_LAYERS_COLOR)
#
#mlp_util_file = cog_util.get_generated_file_name(
#  "renderer_mlp_util",
#  N_LAYERS_TRUNK,
#  N_LAYERS_OPACITY,
#  N_LAYERS_COLOR,
#)
#if N_LAYERS_TRUNK > 0:
#  cog.outl(f"from lightplane.triton_src.generated.{mlp_util_file} import load_mlp_params, mlp_trunk, mlp_opacity, mlp_color")
#else:
#  cog.outl(f"from lightplane.triton_src.generated.{mlp_util_file} import load_mlp_params, mlp_trunk, mlp_opacity, mlp_color")
#]]]
#[[[end]]]
#fmt: on


@triton.jit
def fw_kernel(
    # ---- output -----
    negative_log_transmittance,
    expected_depth,
    expected_features,
    # ---- grid ----
    feature_grid,
    feature_grid_sizes,
    color_feature_grid,
    color_feature_grid_sizes,
    # ----- non-differentiable tensors
    directions,
    origins,
    grid_idx,
    near,
    far,
    rays_encoding,
    inject_noise_seed,
    scaffold,
    # ---- mlp params ----
    mlp_params,  # master ptr for the mlp params
    DIM_HIDDEN_TRUNK: tl.constexpr,
    DIM_HIDDEN_OPACITY: tl.constexpr,
    DIM_HIDDEN_COLOR: tl.constexpr,
    DIM_IN_TRUNK: tl.constexpr,
    DIM_IN_OPACITY: tl.constexpr,
    DIM_IN_COLOR: tl.constexpr,
    DIM_OUT_TRUNK: tl.constexpr,
    DIM_OUT_COLOR: tl.constexpr,
    # ----- config keys ----
    num_samples: tl.constexpr,
    num_samples_inf: tl.constexpr,
    gain: tl.constexpr,
    # ----- sizes ----
    num_rays: tl.constexpr,
    C: tl.constexpr,
    NUM_GRIDS: tl.constexpr,
    NUM_COLOR_GRIDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    # ---- switches ----
    mask_out_of_bounds_samples: tl.constexpr,
    inject_noise: tl.constexpr,
    inject_noise_sigma: tl.constexpr,
    contract_coords: tl.constexpr,
    disparity_at_inf: tl.constexpr,
    use_scaffold: tl.constexpr,
    use_separate_color_grid: tl.constexpr,
):
    # --- init fun called for both fw and bw
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
        seed_buffer,
        sample_index_buffer,
        rays_encoding_buffer,
        one_scaffold,
        zero_value,
        one_vec,
        zero_color,
    ) = fwbw_init(
        directions,
        origins,
        grid_idx,
        near,
        far,
        rays_encoding,
        inject_noise_seed,
        DIM_IN_COLOR,
        DIM_OUT_COLOR,
        num_samples,
        num_samples_inf,
        num_rays,
        C,
        BLOCK_SIZE,
    )

    # delta = (far_buffer - near_buffer) / (num_samples - 1)
    depth = near_buffer

    expected_depth_buffer = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    expected_features_buffer = tl.zeros((BLOCK_SIZE, DIM_OUT_COLOR), dtype=tl.float32)
    prev_transmittance = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    negative_log_transmittance_buffer = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    (
        # [[[cog
        # cog.outl("            " + wb_str_trunk + ("," if len(wb_str_trunk) > 0 else ""))
        # cog.outl("            " + wb_str_opacity + ",")
        # cog.outl("            " + wb_str_color + ",")
        # ]]]
        # [[[end]]]
    ) = load_mlp_params(
        mlp_params,  # master ptr for the mlp params
        DIM_HIDDEN_TRUNK,
        DIM_HIDDEN_OPACITY,
        DIM_HIDDEN_COLOR,
        DIM_IN_TRUNK,
        DIM_IN_OPACITY,
        DIM_IN_COLOR,
        DIM_OUT_TRUNK,
        1,  # =DIM_OUT_OPACITY=1
        DIM_OUT_COLOR,
        BLOCK_SIZE,
    )

    # tl.printf("w0_trunk ", w0_trunk)
    # tl.printf("b0_trunk ", b0_trunk)

    transmittance = tl.exp(-negative_log_transmittance_buffer)

    for step in range(tot_num_samples):
        if step < num_samples:
            depth = depth_lin(near_buffer, far_buffer, num_samples, step)
            depth_prev = depth_lin(near_buffer, far_buffer, num_samples, step - 1)
        else:
            depth = depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                step - num_samples,
            )
            depth_prev = depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                step - num_samples - 1,
            )
        delta = depth - depth_prev

        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z
        if contract_coords:
            sample_x, sample_y, sample_z = contract_pi(sample_x, sample_y, sample_z)

        if use_scaffold:
            scaffold_mask = voxel_grid_sample_one_nearest(
                NUM_GRIDS,
                scaffold,
                feature_grid_sizes,
                grid_idx_buffer,
                sample_x,
                sample_y,
                sample_z,
                1,
                BLOCK_SIZE,
                1,
            )
            scaffold_mask = tl.view(scaffold_mask, (BLOCK_SIZE,))

        else:
            scaffold_mask = one_scaffold

        if tl.sum(scaffold_mask, axis=0):
            # at least one sampled scaffold entry is active so we eval the mlp
            sampled = sample_grid_rep(
                feature_grid,
                feature_grid_sizes,
                grid_idx_buffer,
                sample_x,
                sample_y,
                sample_z,
                C,
                NUM_GRIDS,
                BLOCK_SIZE,
                mask_out_of_bounds_samples,
            )

            if use_separate_color_grid:
                trunk_feature = tl.maximum(sampled, 0.0)

            else:
                # mlp
                trunk_feature = mlp_trunk(
                    sampled,
                    # [[[cog
                    # cog.outl("        " + wb_str_trunk + ("," if len(wb_str_trunk) > 0 else ""))
                    # ]]]
                    # [[[end]]]
                )

            # final opacity value
            opacity_raw = mlp_opacity(
                trunk_feature,
                # [[[cog
                # cog.outl("        " + wb_str_opacity + ",")
                # ]]]
                # [[[end]]]
            )

            if inject_noise:
                r = int_to_randn(
                    sample_index_buffer,
                    sample_index_buffer + num_rays * tot_num_samples,
                    seed_buffer,
                )
                inject_opacity_noise = r * inject_noise_sigma
                opacity_raw = opacity_raw + inject_opacity_noise

            opacity = softplus(opacity_raw)
            delta_opacity = delta * gain * opacity

            if use_separate_color_grid:
                trunk_feature_color = sample_grid_rep(
                    color_feature_grid,
                    color_feature_grid_sizes,
                    grid_idx_buffer,
                    sample_x,
                    sample_y,
                    sample_z,
                    C,
                    NUM_COLOR_GRIDS,
                    BLOCK_SIZE,
                    mask_out_of_bounds_samples,
                )
                trunk_feature_color = tl.maximum(trunk_feature_color, 0.0)
            else:
                trunk_feature_color = trunk_feature

            trunk_feature_and_ray = trunk_feature_color + rays_encoding_buffer

            # if step==0:
            #     tl.printf("vec_and_buf ", vec_and_buf)
            log_color = mlp_color(
                trunk_feature_and_ray,
                # [[[cog
                # cog.outl("        " + wb_str_color + ",")
                # ]]]
                # [[[end]]]
            )

            # if step==0:
            #     tl.printf("log_color ", log_color)

            color = tl.sigmoid(log_color)

            # we must re-mask the values with scaffold here
            delta_opacity = delta_opacity * scaffold_mask
            color = color * tl.view(scaffold_mask[:, None], (BLOCK_SIZE, 1))

        else:
            # Scaffold yields 0 -> we render 0 colors/opacity.
            delta_opacity = zero_value
            color = zero_color
            # tl.printf("skipping samples ", step)

        # neg log transmittance
        negative_log_transmittance_buffer = (
            negative_log_transmittance_buffer + delta_opacity
        )
        transmittance = tl.exp(-negative_log_transmittance_buffer)
        render_weights = prev_transmittance - transmittance

        # exp depth
        expected_depth_buffer = expected_depth_buffer + render_weights * depth

        # render weights
        render_weights = prev_transmittance - transmittance
        render_weights_bcast = tl.view(render_weights[:, None], (BLOCK_SIZE, 1))

        feature_render = color * render_weights_bcast

        expected_features_buffer += feature_render
        prev_transmittance = transmittance
        sample_index_buffer = sample_index_buffer + 1

    tl.store(
        negative_log_transmittance + offs,
        negative_log_transmittance_buffer,
        mask=offs_mask,
    )
    tl.store(expected_depth + offs, expected_depth_buffer, mask=offs_mask)
    tl.store(
        expected_features + offs_features,
        expected_features_buffer,
        mask=offs_features_mask,
    )
