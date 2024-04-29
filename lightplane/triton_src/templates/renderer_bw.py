# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import triton
import triton.language as tl

from lightplane.triton_src.shared.func_util import d_sigmoid, d_softplus, softplus
from lightplane.triton_src.shared.fwbw_util import fwbw_init
from lightplane.triton_src.shared.grid_sample_util import (
    is_in_bounds,
    sample_grid_rep,
    splat_grid_rep,
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
#N_LAYERS_TRUNK, N_LAYERS_OPACITY, N_LAYERS_COLOR = int(N_LAYERS_TRUNK), int(N_LAYERS_OPACITY), int(N_LAYERS_COLOR)
#cog_util = importlib.import_module(COG_UTIL_MODULE)
#mlp_util = importlib.import_module(MLP_UTIL_MODULE)
#
## these strings are needed later:
#wb_str_trunk = cog_util.get_wb_str("trunk", N_LAYERS_TRUNK)
#wb_str_opacity = cog_util.get_wb_str("opacity", N_LAYERS_OPACITY)
#wb_str_color = cog_util.get_wb_str("color", N_LAYERS_COLOR)
#
#xwb_str_trunk = cog_util.get_xwb_str("trunk", N_LAYERS_TRUNK)
#xwb_str_opacity = cog_util.get_xwb_str("opacity", N_LAYERS_OPACITY)
#xwb_str_color = cog_util.get_xwb_str("color", N_LAYERS_COLOR)
#
#dwb_str_trunk = cog_util.get_dwb_str("trunk", N_LAYERS_TRUNK)
#dwb_str_opacity = cog_util.get_dwb_str("opacity", N_LAYERS_OPACITY)
#dwb_str_color = cog_util.get_dwb_str("color", N_LAYERS_COLOR)
#
#x_str_trunk = cog_util.get_x_str("trunk", N_LAYERS_TRUNK)
#x_str_opacity = cog_util.get_x_str("opacity", N_LAYERS_OPACITY)
#x_str_color = cog_util.get_x_str("color", N_LAYERS_COLOR)
#
#mlp_util_file = cog_util.get_generated_file_name(
#    "renderer_mlp_util",
#    N_LAYERS_TRUNK,
#    N_LAYERS_OPACITY,
#    N_LAYERS_COLOR,
#)
#if N_LAYERS_TRUNK > 0:
#   cog.outl(f"from  lightplane.triton_src.generated.{mlp_util_file} import load_mlp_params, mlp_trunk_with_inter_feat, mlp_opacity_with_inter_feat, mlp_color_with_inter_feat, d_mlp_trunk, d_mlp_opacity, d_mlp_color, update_mlp_params, init_mlp_params_grads, update_mlp_params")
#else:
#   cog.outl(f"from  lightplane.triton_src.generated.{mlp_util_file} import load_mlp_params, mlp_opacity_with_inter_feat, mlp_color_with_inter_feat, d_mlp_opacity, d_mlp_color, update_mlp_params, init_mlp_params_grads, update_mlp_params")
#]]]
#[[[end]]]
#fmt: on


@triton.jit
def bw_kernel(
    # ---- output -----
    negative_log_transmittance,
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
    # ----- gradients input-----
    grad_expected_depth,
    grad_negative_log_transmittance,
    grad_expected_features,
    # ----- gradients output-----
    grad_feature_grid,
    grad_color_feature_grid,
    grad_mlp_params,
    grad_rays_enc,
    debug_tensor,
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

    # we count samples from the top for bw
    sample_index_buffer += tot_num_samples - 1

    # delta = (far_buffer - near_buffer) / (num_samples - 1)
    depth = far_buffer

    prev_transmittance = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)

    (
        # [[[cog
        # cog.outl("        " + wb_str_trunk + ("," if len(wb_str_trunk) > 0 else ""))
        # cog.outl("        " + wb_str_opacity + ",")
        # cog.outl("        " + wb_str_color + ",")
        # ]]]
        # [[[end]]]
    ) = load_mlp_params(
        mlp_params,  # master ptr for the mlp params
        DIM_HIDDEN_TRUNK,
        DIM_HIDDEN_OPACITY,
        DIM_HIDDEN_COLOR,
        C,
        DIM_IN_OPACITY,
        DIM_IN_COLOR,
        DIM_OUT_TRUNK,
        1,
        DIM_OUT_COLOR,
        BLOCK_SIZE,
    )

    (
        # grad weight buffers, generated by cog
        # [[[cog
        # for fn in (cog_util.get_dwb_str, cog_util.get_zerowb_str):
        #   for mlp_name, n_layers in zip(("TRUNK", "OPACITY", "COLOR"), (N_LAYERS_TRUNK, N_LAYERS_OPACITY,N_LAYERS_COLOR)):
        #       if n_layers<=0:
        #           continue
        #       value_str = fn(mlp_name, n_layers)
        #       cog.outl(f" {value_str},")
        # ]]]
        # [[[end]]]
    ) = init_mlp_params_grads(
        DIM_HIDDEN_TRUNK,
        DIM_HIDDEN_OPACITY,
        DIM_HIDDEN_COLOR,
        C,
        DIM_IN_OPACITY,
        DIM_IN_COLOR,
        DIM_OUT_TRUNK,
        1,
        DIM_OUT_COLOR,
    )
    d_rays_enc = tl.zeros((BLOCK_SIZE, DIM_IN_COLOR), dtype=tl.float32)
    d_rays_enc_zero = tl.zeros((BLOCK_SIZE, DIM_IN_COLOR), dtype=tl.float32)

    # input grad buffers
    grad_negative_log_transmittance_buffer = tl.load(
        grad_negative_log_transmittance + offs, mask=offs_mask, other=0.0
    ).to(tl.float32)
    grad_expected_features_buffer = tl.load(
        grad_expected_features + offs_features, mask=offs_features_mask, other=0.0
    ).to(tl.float32)
    grad_expected_depth_buffer = tl.load(
        grad_expected_depth + offs, mask=offs_mask, other=0.0
    ).to(tl.float32)

    # intermediate buffers
    prev_proj_depth = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    prev_proj_features = tl.zeros((BLOCK_SIZE, DIM_OUT_COLOR), dtype=tl.float32)
    negative_log_transmittance_buffer = tl.load(
        negative_log_transmittance + offs, mask=offs_mask, other=0.0
    ).to(tl.float32)
    transmittance = tl.exp(-negative_log_transmittance_buffer)
    prev_grad_opacity = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    prev_transmittance = transmittance
    for step in range(tot_num_samples):
        if step < num_samples_inf:
            depth = depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                num_samples_inf - step - 1,
            )
            depth_prev = depth_inv_sphere(
                far_buffer,
                disparity_at_inf,
                num_samples_inf,
                num_samples_inf - step - 2,
            )
        else:
            depth = depth_lin(
                near_buffer,
                far_buffer,
                num_samples,
                num_samples - (step - num_samples_inf) - 1,
            )
            depth_prev = depth_lin(
                near_buffer,
                far_buffer,
                num_samples,
                num_samples - (step - num_samples_inf) - 2,
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

        scaffold_mask_unsqueeze = tl.view(scaffold_mask[:, None], (BLOCK_SIZE, 1))

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
                # we use relufields
                trunk_feature = tl.maximum(sampled, 0.0)

            else:
                pass
                # trunk mlp
                # [[[cog
                # if N_LAYERS_TRUNK > 0:
                #   cog.outl(f"(trunk_feature, " + x_str_trunk + ", " + xwb_str_trunk + " ) = mlp_trunk_with_inter_feat(")
                #   cog.outl("    sampled,")
                #   cog.outl("    "+ wb_str_trunk + ",")
                #   cog.outl(")")
                # else:
                #   cog.outl("pass")
                # cog.outl("")
                # ]]]
                # [[[end]]]

            # opacity mlp
            # [[[cog
            # cog.outl("# final opacity value")
            # if N_LAYERS_OPACITY > 1:
            #    cog.outl(f"(opacity_raw, " + x_str_opacity + ", " + xwb_str_opacity + " ) = mlp_opacity_with_inter_feat(")
            # else:
            #    cog.outl(f"(opacity_raw, " + x_str_opacity + " ) = mlp_opacity_with_inter_feat(")
            # cog.outl("    trunk_feature,")
            # cog.outl("    "+ wb_str_opacity + ",")
            # cog.outl(")")
            # ]]]
            # [[[end]]]

            if inject_noise:
                r = int_to_randn(
                    sample_index_buffer,
                    sample_index_buffer + num_rays * tot_num_samples,
                    seed_buffer,
                )
                inject_opacity_noise = r * inject_noise_sigma
                opacity_raw = opacity_raw + inject_opacity_noise

            opacity = softplus(opacity_raw) * scaffold_mask
            delta_opacity = delta * gain * opacity

            if use_separate_color_grid:
                sample_trunk_feature_color = sample_grid_rep(
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
                trunk_feature_color = tl.maximum(sample_trunk_feature_color, 0.0)
            else:
                trunk_feature_color = trunk_feature

            trunk_feature_and_ray = trunk_feature_color + rays_encoding_buffer

            # [[[cog
            # if N_LAYERS_COLOR > 1:
            #    cog.outl(f"(log_color, " + x_str_color + ", " + xwb_str_color + " ) = mlp_color_with_inter_feat(")
            # else:
            #    cog.outl(f"(log_color, " + x_str_color + " ) = mlp_color_with_inter_feat(")
            # cog.outl("    trunk_feature_and_ray,")
            # cog.outl("    "+ wb_str_color + ",")
            # cog.outl(")")
            # ]]]
            # [[[end]]]

            color = tl.sigmoid(log_color)

            # we must re-mask the values with scaffold here
            delta_opacity = delta_opacity * scaffold_mask
            color = color * scaffold_mask_unsqueeze

            # grads
            proj_features = (
                color * grad_expected_features_buffer * scaffold_mask_unsqueeze
            )
            proj_depth = depth * grad_expected_depth_buffer * scaffold_mask

            prev_transmittance = transmittance

            opacity_grad_now = prev_transmittance * (
                (proj_depth - prev_proj_depth)
                + tl.sum(proj_features - prev_proj_features, axis=1)
            )

            prev_grad_opacity += opacity_grad_now

            # update to the transmittance of the prev step
            negative_log_transmittance_buffer = (
                negative_log_transmittance_buffer - delta_opacity
            )

            transmittance = tl.exp(-negative_log_transmittance_buffer)

            grad_opacity = (
                delta
                * (prev_grad_opacity + grad_negative_log_transmittance_buffer)
                * scaffold_mask
            )

            grad_opacity_raw = gain * d_softplus(grad_opacity, opacity_raw)
            grad_opacity_raw = tl.view(grad_opacity_raw[:, None], (BLOCK_SIZE, 1))

            # grad opacity head
            # [[[cog
            # cog.outl(f"d_trunk_opacity, "+ dwb_str_opacity + "  = d_mlp_opacity(grad_opacity_raw, ")
            # if N_LAYERS_OPACITY > 1:
            #   cog.outl(f"       "+ wb_str_opacity +", "+ xwb_str_opacity +", "+ x_str_opacity +")")
            # else:
            #   cog.outl(f"       "+ wb_str_opacity +", "+ x_str_opacity +")")
            # dim_last_opacity = "DIM_HIDDEN_OPACITY" if N_LAYERS_OPACITY > 1 else "DIM_IN_OPACITY"
            # cog.outl(f"dw{N_LAYERS_OPACITY-1}_opacity = tl.view(dw{N_LAYERS_OPACITY-1}_opacity, (1, {dim_last_opacity}))")
            # ]]]
            # [[[end]]]

            transmittance_diff = transmittance - prev_transmittance
            transmittance_diff = tl.view(transmittance_diff[:, None], (BLOCK_SIZE, 1))

            # add the feature grad again
            d_color = transmittance_diff * tl.view(
                grad_expected_features_buffer, (BLOCK_SIZE, DIM_OUT_COLOR)
            )

            d_log_color = d_sigmoid(d_color, log_color)

            # [[[cog
            # cog.outl(f"d_trunk_color, "+ dwb_str_color + "  = d_mlp_color(d_log_color, ")
            # if N_LAYERS_COLOR > 1:
            #   cog.outl(f"       "+ wb_str_color +", "+ xwb_str_color +", "+ x_str_color +")")
            # else:
            #   cog.outl(f"       "+ wb_str_color +", "+ x_str_color +")")
            # ]]]
            # [[[end]]]

            d_rays_enc_ = d_trunk_color

            if use_separate_color_grid:
                # we use relufields
                d_trunk_color_relu = d_trunk_color * (trunk_feature_color > 0.0).to(
                    tl.float32
                )
                d_trunk_opacity *= (trunk_feature > 0.0).to(tl.float32)

                splat_grid_rep(
                    d_trunk_opacity,
                    grad_feature_grid,
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

                splat_grid_rep(
                    d_trunk_color_relu,
                    grad_color_feature_grid,
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

            else:
                d_trunk = d_trunk_color + d_trunk_opacity
                # [[[cog
                # if N_LAYERS_TRUNK > 0:
                #   cog.outl(f"d_sampled, "+ dwb_str_trunk + "  = d_mlp_trunk(d_trunk, ")
                #   cog.outl(f"   "+ wb_str_trunk +", "+ xwb_str_trunk +", "+ x_str_trunk +")")
                # ]]]
                # [[[end]]]
                # grad MLP

                splat_grid_rep(
                    d_sampled,
                    grad_feature_grid,
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

        # ----- if scaffold_mask_sum: else part
        else:
            # Scaffold yields 0 -> we render 0 colors/opacity.
            value = zero_value
            color = zero_color
            delta_value = zero_value

            # grads
            proj_features = (
                color * grad_expected_features_buffer * scaffold_mask[:, None]
            )
            proj_depth = depth * grad_expected_depth_buffer * scaffold_mask

            prev_transmittance = transmittance

            opacity_grad_now = prev_transmittance * (
                (proj_depth - prev_proj_depth)
                + tl.sum(proj_features - prev_proj_features, axis=1)
            )

            prev_grad_opacity = prev_grad_opacity + opacity_grad_now

            # update to the transmittance of the prev step
            negative_log_transmittance_buffer = (
                negative_log_transmittance_buffer - delta_value
            )

            transmittance = tl.exp(-negative_log_transmittance_buffer)

            # [[[cog
            # def create_grad_value_function(mlp_name, n_layers):
            #   for l in range(n_layers):
            #       dim_out = f"DIM_OUT_{mlp_name}" if (l == int(n_layers) - 1) else f"DIM_HIDDEN_{mlp_name}"
            #       cog.outl(f"dw{l}_{mlp_name} = zero_w{l}_{mlp_name.upper()}")
            #       cog.outl(f"db{l}_{mlp_name} = zero_b{l}_{mlp_name.upper()}")
            #
            # create_grad_value_function("trunk", N_LAYERS_TRUNK)
            # create_grad_value_function("opacity", N_LAYERS_OPACITY)
            # create_grad_value_function("color", N_LAYERS_COLOR)
            # ]]]
            # [[[end]]]
            d_rays_enc_ = d_rays_enc_zero

        # [[[cog
        # def create_grad_accum_function(mlp_name, MLP_NAME, n_layers):
        #   for l in range(n_layers):
        #       dim_out = f"DIM_OUT_{mlp_name}" if (l == int(n_layers) - 1) else f"DIM_HIDDEN_{mlp_name}"
        #       cog.outl(f"dw{l}_{mlp_name} += dw{l}_{MLP_NAME}")
        #       cog.outl(f"db{l}_{mlp_name} += db{l}_{MLP_NAME}")
        #
        # create_grad_accum_function("TRUNK", "trunk", N_LAYERS_TRUNK)
        # create_grad_accum_function("OPACITY", "opacity", N_LAYERS_OPACITY)
        # create_grad_accum_function("COLOR", "color", N_LAYERS_COLOR)
        # ]]]
        # [[[end]]]
        d_rays_enc += d_rays_enc_
        prev_proj_depth = proj_depth
        prev_proj_features = proj_features
        sample_index_buffer = sample_index_buffer - 1

    ## update the weight, bias grads
    update_mlp_params(
        grad_mlp_params,  # master ptr for the mlp params
        DIM_HIDDEN_TRUNK,
        DIM_HIDDEN_OPACITY,
        DIM_HIDDEN_COLOR,
        C,
        DIM_IN_OPACITY,
        DIM_IN_COLOR,
        DIM_OUT_TRUNK,
        1,
        DIM_OUT_COLOR,
        # [[[cog
        # trunk = cog_util.get_dwb_str("TRUNK", N_LAYERS_TRUNK)
        # opacity = cog_util.get_dwb_str("OPACITY", N_LAYERS_OPACITY)
        # color = cog_util.get_dwb_str("COLOR", N_LAYERS_COLOR)
        # if N_LAYERS_TRUNK > 0:
        #   cog.outl(f"     {trunk},")
        # cog.outl(f"     {opacity},")
        # cog.outl(f"     {color},")
        # ]]]
        # [[[end]]]
    )

    tl.store(
        grad_rays_enc
        + pid * BLOCK_SIZE * DIM_IN_COLOR
        + DIM_IN_COLOR * tl.arange(0, BLOCK_SIZE)[:, None]
        + tl.arange(0, DIM_IN_COLOR)[None, :],
        d_rays_enc,
        mask=offs_features_mask,
    )
