# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from typing import Any, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


from .lightplane_renderer import lightplane_renderer
from .misc_utils import if_not_none_else
from .mlp_utils import (
    DecoderParams,
    init_decoder_params,
    flattened_decoder_params_to_list,
)
from .naive_renderer import (
    lightplane_eval_mlp,
    lightplane_eval_mlp_opacity_only,
    lightplane_renderer_naive,
)
from .ray_utils import (
    Rays,
    calc_harmonic_embedding,
    calc_harmonic_embedding_dim,
    jitter_near_far,
)


class LightplaneRenderer(torch.nn.Module):
    def __init__(
        self,
        num_samples: int,
        color_chn: int,
        grid_chn: int,
        mlp_hidden_chn: int,
        mlp_n_layers_opacity: int = 2,
        mlp_n_layers_trunk: int = 2,
        mlp_n_layers_color: int = 2,
        use_separate_color_grid: bool = False,
        opacity_init_bias: float = -5.0,
        gain: float = 1.0,
        bg_color: tuple[float, ...] | float = 0.0,
        enable_direction_dependent_colors: bool = True,
        ray_embedding_num_harmonics: int | None = 3,
        num_samples_inf: int = 0,
        mask_out_of_bounds_samples: bool = False,
        contract_coords: bool = False,
        disparity_at_inf: float = 1e-5,
        inject_noise_sigma: float = 0.0,
        inject_noise_seed: int | None = None,
        rays_jitter_near_far: bool = False,
        return_log_transmittance: bool = False,
        triton_block_size: int = 16,
        triton_num_warps: int = 4,
        use_naive_impl: bool = False,
    ) -> None:
        r"""
        This is the Pytorch Module for the Lightplane Renderer.
        It uses `lightplane_renderer` as the core function for rendering and
        automatically initialize the parameters for the MLPs used in the renderer.

        Args:
            num_samples: Number of samples to render.
            color_chn: Number of channels for the rendererd color.
            grid_chn: Number of channels for the 3D grid to be rendered.
            mlp_hidden_chn: Number of hidden channels for all MLPs.
            mlp_n_layers_opacity: Number of layers for `opacity_mlp`.
            mlp_n_layers_trunk: Number of layers for `trunk_mlp`.
            mlp_n_layers_color: Number of layers for `color_mlp`.
            use_separate_color_grid: Whether using a separate grid-list for
                colors.
            opacity_init_bias: Initial bias for `opacity_mlp`.
            gain: `gain` for the `lightplane_renderer`.
            bg_color: Background color.
            enable_direction_dependent_colors: Enable ray-direction dependent
                rendering.
            ray_embedding_num_harmonics: Level of harmonic functions for
                ray-direction embedding.
                Setting ray_embedding_num_harmonics=0 will only use the
                ray direction as the ray encoding.
                A value bigger than 0 will append the harmonic embedding to the
                ray direction.
                Finally, setting ray_embedding_num_harmonics=None will use the
                ray `encoding` field from the `Rays` object input to forward.
            num_samples_inf: Number of samples beyond the far plane.
            mask_out_of_bounds_samples: Whether to mask out-of-bounds samples.
            contract_coords: Whether to contract the coordinates as in MeRF.
            disparity_at_inf: The beyond-far samples (their number-per-ray
                is determined by `num_samples_inf`) are sampled in the disparity space
                in `range[far, 1 / disparity_at_inf]`.
            inject_noise_sigma: Standard deviation of the opacity noise to inject.
            inject_noise_seed: Seed for the opacity noise to inject.
            rays_jitter_near_far: Whether to jitter the near and far planes
                uniformly in range `[-delta, delta]`.
            return_log_transmittance: Whether to return the log transmittance
                instead of the `[0, 1]` alpha mask.
            use_naive_impl: Whether to use the naive pytorch implementation
                instead of triton.
            triton_block_size: Block size for triton.
            triton_num_warps: Number of warps for triton.
        """

        super().__init__()

        self.num_samples = num_samples
        self.opacity_init_bias = opacity_init_bias
        self.gain = gain
        self.color_chn = color_chn
        self.num_samples_inf = num_samples_inf
        self.mask_out_of_bounds_samples = mask_out_of_bounds_samples
        self.contract_coords = contract_coords
        self.disparity_at_inf = disparity_at_inf
        self.inject_noise_sigma = inject_noise_sigma
        self.inject_noise_seed = inject_noise_seed
        self.rays_jitter_near_far = rays_jitter_near_far
        self.triton_block_size = triton_block_size
        self.triton_num_warps = triton_num_warps
        self.use_naive_impl = use_naive_impl
        self.return_log_transmittance = return_log_transmittance
        self.enable_direction_dependent_colors = enable_direction_dependent_colors
        self.ray_embedding_num_harmonics = ray_embedding_num_harmonics

        if use_separate_color_grid:
            if mlp_n_layers_trunk > 0:
                logger.warning(
                    "Auto-setting mlp_n_layers_trunk=0 because a separate feature grid"
                    " for colors is used (use_separate_color_grid=True)."
                )
            mlp_n_layers_trunk = 0

        decoder_params = init_decoder_params(
            device="cpu",
            n_layers_opacity=mlp_n_layers_opacity,
            n_layers_trunk=mlp_n_layers_trunk,
            n_layers_color=mlp_n_layers_color,
            input_chn=grid_chn,
            hidden_chn=mlp_hidden_chn,
            color_chn=color_chn,
            opacity_init_bias=opacity_init_bias,
            pad_color_channels_to_min_block_size=True,
            use_separate_color_grid=use_separate_color_grid,
        )
        # ray encoding dim is the same as the input channels for the color MLP
        self.rays_encoding_dim = decoder_params.n_hidden_color[0]

        self.mlp_params = torch.nn.Parameter(decoder_params.mlp_params)

        if self.ray_embedding_num_harmonics is not None:
            if not self.enable_direction_dependent_colors:
                raise ValueError(
                    "LightplaneRenderer's viewpoint dependent colors are disabled,"
                    " (enable_direction_dependent_colors=False), but"
                    " `ray_embedding_num_harmonics` is set. Set"
                    " LightplaneRender.ray_embedding_num_harmonics = None"
                    " if you intended to disable viewpoint dependent colors."
                )
            # initialize the embedder of ray directions
            color_branch_input_chn = decoder_params.n_hidden_color[0].item()
            self.harmonic_ray_embedding_linear = torch.nn.Linear(
                calc_harmonic_embedding_dim(self.ray_embedding_num_harmonics),
                color_branch_input_chn,
            )
            # xavier init of the ray encoding linear layer
            torch.nn.init.xavier_uniform_(self.harmonic_ray_embedding_linear.weight)
            self.harmonic_ray_embedding_linear.bias.data.fill_(0.0)
            
        # register the n_hidden_XXX fields of decoder_params
        for field_name, fields in asdict(decoder_params).items():
            if field_name != "mlp_params" and isinstance(fields, torch.Tensor):
                self.register_buffer(field_name, fields, persistent=False)

        self.register_buffer("bg_color", self._process_bg_color(bg_color))

    def eval_decoder_at_points(
        self,
        pts: torch.Tensor,
        pts_to_grid_idx: torch.Tensor,
        rays_encoding: torch.Tensor | None,
        feature_grid: tuple[torch.Tensor, ...],
        color_feature_grid: tuple[torch.Tensor, ...] | None = None,
        scaffold: torch.Tensor | None = None,
        # If set, the following args override the module's default values:
        gain: float | None = None,
        mask_out_of_bounds_samples: bool | None = None,
        contract_coords: bool | None = None,
        directions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Evaluates the MLP at the given points.

        Args:
            pts: Points to evaluate the MLP at, shape `[n_rays, n_pts, 3]`.
            pts_to_grid_idx: The grid index of each point, long tensor of shape
                `[n_pts, ]`.
            rays_encoding: Ray encoding for each point.
            feature_grid: feature grid.
            color_feature_grid: color feature grid.
            scaffold: `scaffold` for the MLP.
            gain:  `gain` for the `lightplane_renderer`.
            mask_out_of_bounds_samples: Whether to mask out-of-bounds samples.
            contract_coords: Whether to contract the coordinates as in MeRF.
            directions: Per-ray viewing directions `[n_rays, 3]`;
                can be passed instead of `rays_encoding`.
        """

        n_rays, n_pts, pts_dim = pts.shape
        assert pts_dim == 3
        assert pts_to_grid_idx.shape == (n_rays,)
        if rays_encoding is not None:
            assert rays_encoding.shape == [n_rays, self.rays_encoding_dim]
        else:
            assert (
                directions is not None
            ), "Must pass one of (rays_encoding, directions)"
            assert directions.shape == (n_rays, 3)

        return lightplane_eval_mlp(
            points=pts,
            grid=feature_grid,
            ray_grid_idx=pts_to_grid_idx,
            decoder_params=self.get_decoder_params(),
            rays_encoding=self._get_ray_encoding(rays_encoding, directions),
            gain=if_not_none_else(gain, self.gain),
            contract_coords=if_not_none_else(contract_coords, self.contract_coords),
            mask_out_of_bounds_samples=if_not_none_else(
                mask_out_of_bounds_samples,
                self.mask_out_of_bounds_samples,
            ),
            inject_opacity_noise=None,
            scaffold=scaffold,
            color_grid=color_feature_grid,
        )

    def get_decoder_params(self) -> DecoderParams:
        r"""
        Helper function to get the `DecoderParams` object from the module.
        """
        return DecoderParams(
            self.mlp_params,
            self.n_hidden_trunk,
            self.n_hidden_opacity,
            self.n_hidden_color,
            color_chn=self.color_chn,
        )
        
    def get_decoder_params_list(self) -> Tuple[
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, ...],
    ]:
        r"""
        Helper function to get the list of weight matrices and bias vectors
        of the MLPs in the renderer's decoder:
        
        Returns:
            weights_trunk: Weight matrices of the trunk MLP.
            biases_trunk: Bias vectors of the trunk MLP.
            weights_opacity: Weight matrices of the opacity MLP.
            biases_opacity: Bias vectors of the opacity MLP.
            weights_color: Weight matrices of the color MLP.
            biases_color: Bias vectors of the color MLP.
        """
        return flattened_decoder_params_to_list(
            self.mlp_params,
            self.n_hidden_trunk,
            self.n_hidden_opacity,
            self.n_hidden_color,
        )

    def _process_bg_color(
        self, bg_color: tuple[float, ...] | float | None
    ) -> torch.Tensor:
        r"""
        Helper function to process the background color.

        Args:
            bg_color: Background color.
        """
        if bg_color is None:
            return self.bg_color
        if isinstance(bg_color, float):
            bg_color = torch.tensor([bg_color] * self.color_chn, dtype=torch.float)
        elif torch.is_tensor(bg_color):
            pass
        else:
            bg_color = torch.tensor(bg_color, dtype=torch.float)
        assert len(bg_color) == self.color_chn
        return bg_color

    def eval_opacity_at_points(
        self,
        pts: torch.Tensor,
        pts_to_grid_idx: torch.Tensor,
        feature_grid: tuple[torch.Tensor, ...] | torch.Tensor,
        scaffold: torch.Tensor | None = None,
        # If set, the following args override the module's default values:
        gain: float | None = None,
        mask_out_of_bounds_samples: bool | None = None,
        grid_sizes: list[list[int]] | None = None,
    ):
        """
        Calcualte the opacities at the given points.

        Args:
            pts: Points to evaluate at, shape `[n_rays, n_pts, 3]`.
            pts_to_grid_idx: The grid index of each point, long tensor of shape
                `[n_pts, ]`.
            feature_grid: Feature grid to render from.
            scaffold: `scaffold` for the MLP.
            gain:  `gain` for the `lightplane_renderer`.
            mask_out_of_bounds_samples: Whether to mask out-of-bounds samples.
            grid_sizes: The size of the `feature_grid`. Only required if
                `feature_grid` is a 2D tensor.
        Returns:
            results: Opacities at the given points. `[n_rays, n_pts]`.
        """
        n_rays, n_pts, pts_dim = pts.shape
        assert pts_dim == 3
        assert pts_to_grid_idx.shape == (n_rays,)

        results = lightplane_eval_mlp_opacity_only(
            points=pts,
            grid=feature_grid,
            ray_grid_idx=pts_to_grid_idx,
            decoder_params=self.get_decoder_params(),
            gain=if_not_none_else(gain, self.gain),
            mask_out_of_bounds_samples=if_not_none_else(
                mask_out_of_bounds_samples,
                self.mask_out_of_bounds_samples,
            ),
            inject_opacity_noise=None,
            scaffold=scaffold,
        )
        return results

    @torch.no_grad()
    def calculate_scaffold(
        self,
        feature_grid: tuple[torch.Tensor, ...] | torch.Tensor,
        scaffold_size: tuple[int, int, int, int],  # [B, D, H, W],
        device,
        threshold: float = 1e-7,
        grid_sizes: list[tuple[int, int, int, int, int]] | None = None,
        dilate_scaffold: int = 2,
    ):
        """
        Calculate `scaffold` by sampling voxel grid with the shape of
        `scaffold_size` and prunning points whose opacities are below
        `threshold`.

        Args:
            feature_grid: Feature grid to render from.
            scaffold_size: The shape of `scaffold`.
                `scaffold` should be a voxel grid with the same batch size as
                `feature_grid`.
                Example::

                    scaffold_size = [B, D, H, W]

            device: Tensor device.
            threshold: The opacity threshold to prune points.
            grid_sizes: The size of the `feature_grid`. Only required if
                `feature_grid` is a 2D tensor.
            dilate_scaffold: Dilate opacities before thresholding to increase the
                extent of the occupied regions in the scaffold.
        """
        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, scaffold_size[1]),
                torch.linspace(0, 1, scaffold_size[2]),
                torch.linspace(0, 1, scaffold_size[3]),
            ),
            -1,
        ).to(
            device
        )  # [D, H, W, 3]
        samples = samples.permute(2, 1, 0, 3)
        dense_xyz = samples * 2.0 - 1.0
        scaffold = torch.ones(
            (scaffold_size[0], scaffold_size[1], scaffold_size[2], scaffold_size[3]),
            device=device,
        )  # [B, D, H, W, 3]
        for i in range(scaffold_size[0]):  # B
            for j in range(scaffold_size[1]):
                sampled_results = self.eval_opacity_at_points(
                    pts=dense_xyz[j],
                    pts_to_grid_idx=(
                        torch.ones((scaffold_size[2],), device=device) * i
                    ).to(torch.long),
                    feature_grid=feature_grid,
                    scaffold=None,
                    gain=self.gain,
                    mask_out_of_bounds_samples=self.mask_out_of_bounds_samples,
                )
                scaffold[i, j] = sampled_results

        # use a max_pool filter to dilate the scaffold
        if dilate_scaffold > 0:
            dilation_ks = dilate_scaffold * 2 + 1
            scaffold = torch.nn.functional.max_pool3d(
                scaffold, kernel_size=dilation_ks, padding=dilate_scaffold, stride=1
            )

        scaffold = (scaffold > threshold) * 1.0
        return scaffold

    def forward(
        self,
        rays: Rays,
        feature_grid: tuple[torch.Tensor, ...] | torch.Tensor,
        color_feature_grid: tuple[torch.Tensor, ...] | None = None,
        scaffold: torch.Tensor | None = None,
        grid_sizes: list[list[int]] | None = None,
        color_grid_sizes: list[list[int]] | None = None,
        # If set, the following args override the module's default values:
        bg_color: tuple[float, ...] | float | None = None,
        num_samples: int | None = None,
        gain: float | None = None,
        num_samples_inf: int | None = None,
        mask_out_of_bounds_samples: bool | None = None,
        contract_coords: bool | None = None,
        disparity_at_inf: float | None = None,
        inject_noise_sigma: float | None = None,
        inject_noise_seed: int | None = None,
        rays_jitter_near_far: bool | None = None,
        return_log_transmittance: bool | None = None,
        regenerate_code: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute rendering with Lightplane Renderer.

        Args:
            rays: `Rays` to render.
            feature_grid: Feature grid to render from.
            color_feature_grid: Color feature grid. If provided, the color will be
                rendered based on this grid.
            scaffold: `scaffold` for the MLP.
            grid_sizes: The size of the `feature_grid`. Only required if `feature_grid`
                is a 2D tensor.
            color_grid_sizes: The size of the `color_feature_grid`. Only required if
                `color_feature_grid` is a 2D tensor.
            bg_color: Background color.
            num_samples: Number of samples to render.
            gain: `gain` for the 'lightplane_renderer'.
            num_samples_inf: Number of samples beyond the far plane.
            mask_out_of_bounds_samples: Whether to mask out-of-bounds samples.
            contract_coords: Whether to contract the coordinates as in MeRF.
            disparity_at_inf: The beyond-far samples (their number-per-ray is
                determined by `num_samples_inf`) are sampled in the disparity
                space in `range[far, 1 / disparity_at_inf]`.
            inject_noise_sigma: Standard deviation of the opacity noise to inject.
            inject_noise_seed: Seed for the opacity noise to inject.
            rays_jitter_near_far: Whether to jitter the `near` and `far` planes
                uniformly in range [0, delta].
            return_log_transmittance: Whether to return the log transmittance
                instead of the `[0, 1]` alpha mask.
            regenerate_code: Whether to regenerate the code for the lightplane.

        Returns:
            ray_length_render: The rendered ray length.
            alpha: The alpha mask, either in range `[0, 1]` if
            `return_log_transmittance=False` else real-valued log transmittance.
            feature_render: The rendered feature.
        """

        device = rays.device

        # override the default class variables if input args are set
        bg_color = if_not_none_else(bg_color, self.bg_color)
        num_samples = if_not_none_else(num_samples, self.num_samples)
        gain = if_not_none_else(gain, self.gain)
        num_samples_inf = if_not_none_else(num_samples_inf, self.num_samples_inf)
        mask_out_of_bounds_samples = if_not_none_else(
            mask_out_of_bounds_samples, self.mask_out_of_bounds_samples
        )
        contract_coords = if_not_none_else(contract_coords, self.contract_coords)
        disparity_at_inf = if_not_none_else(disparity_at_inf, self.disparity_at_inf)
        inject_noise_sigma = if_not_none_else(
            inject_noise_sigma, self.inject_noise_sigma
        )
        inject_noise_seed = if_not_none_else(inject_noise_seed, self.inject_noise_seed)
        rays_jitter_near_far = if_not_none_else(
            rays_jitter_near_far, self.rays_jitter_near_far
        )
        return_log_transmittance = if_not_none_else(
            return_log_transmittance, self.return_log_transmittance
        )

        # handle bg color
        bg_color = self._process_bg_color(bg_color).to(device)

        # handle ray encoding
        _check_renderer_ray_encoding_input(
            rays.encoding,
            self.ray_embedding_num_harmonics,
            self.rays_encoding_dim,
            self.enable_direction_dependent_colors,
        )

        rays_with_encoding = copy.copy(rays)  # shallow copy for futher processing
        rays_with_encoding.encoding = self._get_ray_encoding(
            rays.encoding,
            rays.directions,
        )

        if rays_jitter_near_far:
            rays_with_encoding.near, rays_with_encoding.far = jitter_near_far(
                rays_with_encoding.near,
                rays_with_encoding.far,
                num_samples,
            )

        # run lightplane
        lightplane_fn = (
            lightplane_renderer_naive if self.use_naive_impl else lightplane_renderer
        )
        ray_length_render, negative_log_transmittance, feature_render = lightplane_fn(
            rays_with_encoding,
            feature_grid,
            self.get_decoder_params(),
            # ------ config keys ------
            num_samples=num_samples,
            gain=gain,
            num_samples_inf=num_samples_inf,
            mask_out_of_bounds_samples=mask_out_of_bounds_samples,
            contract_coords=contract_coords,
            disparity_at_inf=disparity_at_inf,
            inject_noise_sigma=inject_noise_sigma,
            inject_noise_seed=inject_noise_seed,
            regenerate_code=regenerate_code,
            scaffold=scaffold,
            color_grid=color_feature_grid,
            grid_sizes=grid_sizes,
            color_grid_sizes=color_grid_sizes,
            triton_block_size=self.triton_block_size,
            triton_num_warps=self.triton_num_warps,
        )

        # convert neg log transmittance to alpha mask in range [0, 1]
        inverted_mask = torch.exp(-negative_log_transmittance)

        # apply the bg color
        feature_render = feature_render + inverted_mask[..., None] * bg_color

        # determine whether we return the mask or log-transmittance
        if return_log_transmittance:
            alpha = -negative_log_transmittance
        else:
            alpha = 1 - inverted_mask

        return ray_length_render, alpha, feature_render

    def _get_ray_encoding(
        self, ray_encoding: torch.Tensor | None, directions: torch.Tensor | None
    ) -> torch.Tensor:
        r"""
        Helper function to get the ray encoding.
        """
        if ray_encoding is not None:
            assert not self.enable_direction_dependent_colors
            assert self.ray_embedding_num_harmonics is None
            return ray_encoding

        return self._get_ray_embedding(directions)
    
    def _get_ray_embedding(self, ray_directions: torch.tensor) -> torch.Tensor:
        r"""
        Helper function to get the ray embedding.

        If `self.enable_direction_dependent_colors` is set to `False`,
        we use a zero ray encoding tensor.
        Otherwise, we use the harmonic embedding of the ray directions as the ray
        embedding.

        Ray embedding is used in `color_mlp` to render the color.
        """
        if not self.enable_direction_dependent_colors:
            # if not using ray encoding, we use a zero ray encoding tensor
            return ray_directions.new_zeros(
                ray_directions.shape[0],
                self.rays_encoding_dim,
            )
        else:
            assert self.ray_embedding_num_harmonics is not None
            harmonic_embed = calc_harmonic_embedding(
                torch.nn.functional.normalize(ray_directions, dim=-1),
                self.ray_embedding_num_harmonics,
            )
            return self.harmonic_ray_embedding_linear(harmonic_embed)


def _check_renderer_ray_encoding_input(
    ray_encoding: torch.Tensor | None,
    ray_embedding_num_harmonics: int | None,
    ray_encoding_dim: int,
    enable_direction_dependent_colors: bool,
):
    if ray_encoding is not None and ray_encoding.shape[1] != ray_encoding_dim:
        raise ValueError(
            f"Ray encoding has a wrong dimension."
            f" Expected: {ray_encoding_dim}, got: {ray_encoding.shape[1]}"
        )

    if not enable_direction_dependent_colors:
        if ray_encoding is not None:
            raise ValueError(
                "LightplaneRenderer's viewpoint dependent colors are disabled, "
                " (enable_direction_dependent_colors=False), but the `encoding`"
                " field of `rays` is set. Make sure to set rays.encoding=None"
                " if you intended to disable viewpoint dependent colors."
            )
            
        if ray_embedding_num_harmonics is not None:
            raise ValueError(
                "LightplaneRenderer's viewpoint dependent colors are disabled,"
                " (enable_direction_dependent_colors=False), but the"
                " `ray_embedding_num_harmonics` parameter of `LightplaneRenderer` is set."
                " Make sure to set LightplaneRender.ray_embedding_num_harmonics = None"
                " if you intended to disable viewpoint dependent colors."
            )
            
        return

    if not (
        (ray_embedding_num_harmonics is not None and ray_encoding is not None)
        or (ray_embedding_num_harmonics is None and ray_encoding is None)
    ):
        return

    if ray_encoding is None:
        err_msg = (
            "rays.encoding is unset (=None), but the Lightplane module is"
            " not configured to compute harmonic ray embeddings"
            " (self.ray_embedding_num_harmonics is unset = None)."
        )
    else:
        err_msg = (
            "rays.encoding is set, but the Lightplane module is configured to"
            " aslo compute harmonic ray embeddings"
            " (self.ray_embedding_num_harmonics is set)."
        )

    raise ValueError(
        err_msg
        + (
            " Please chose one of the following: \n"
            " 1) If you wish the lightplane module to compute the ray embeddings,"
            " set self.ray_embedding_num_harmonics to an appropriate integer "
            " and set the `rays.encoding` field of the `rays` object to None."
            " 2) If you wish to use your own ray embeddings (rays.encoding),"
            " set rays.encoding to a [n_rays, ray_encoding_dim] tensor, "
            " and also set the ray_embedding_num_harmonics field of the Lightplane"
            " module to None."
        )
    )
