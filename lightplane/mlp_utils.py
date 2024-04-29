# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import asdict, dataclass, fields
from enum import Enum
from logging import getLogger
from typing import Optional, Tuple

import torch

from .triton_src.shared.const import MIN_BLOCK_SIZE

logger = getLogger(__name__)


@dataclass
class DecoderParams:
    r"""
    Class configuring the learnable parameters of the decoder from Lightplane Renderer.

    The decoder comprises a learnable function that predicts color and an opacity value
    given a grid feature sampled at every point along a rendering ray.

    Specifically, the decoder function consists of three MLPs: `trunk_mlp`, `opacity_mlp`,
    and `color_mlp`.
    The three MLPs predict opacity and color as follows:

        1) `use_separate_color_grid==False`::

            grid -> f_i -> trunk_mlp -> e_i -> e_i + ray_encoding -> color_mlp -> c_i
                                            -> opacity_mlp -> o_i

        If the renderer uses a single grid for both opacity and color, an MLP
        `trunk_mlp` maps the grid-sampled feature `f_i` to a trunk feature `e_i`,
        which is later converted to opacity and color with a pair of additional
        color and opacity MLP heads `color_mlp` and `opacity_mlp`.
        The trunk feature `e_i` is summed with `ray_encoding` before `color_mlp`
        to make the predicted color viewpoint dependent.

        2) `use_separate_color_grid==True`::

            grid       -> f_i  -> opacity_mlp -> o_i
            color_grid -> cf_i -> cf_i + ray_encoding -> color_mlp -> c_i

        If the renderer uses a separate color grid (`use_separate_color_grid==True`),
        the trunk MLP will be omitted
        The `opacity_mlp` and `color_mlp` predict the opacity `o_i` and color
        values `c_i`, respectively,
        given an opacity/color features (`f_i` and `cf_i`) sampled from the
        corresponding grid `grid` and `color_grid`.

    The parameters of the three MLPs are stored in the `mlp_params` attribute.
    Here, `mlp_params` is a 1D tensor which concatenates the flattened weight matrices
    and bias vectors of the three MLPs in the following order::

        mlp_params = torch.cat(
            [
                weights_trunk[0].flatten(),
                ...
                weights_trunk[-1].flatten(),
                biases_trunk[0],
                ...
                biases_trunk[-1],
                weights_opacity[0].flatten(),
                ...
                weights_opacity[-1].flatten(),
                biases_opacity[0],
                ...
                baises_opacity[-1],
                weights_color[0].flatten(),
                ...
                weights_color[n].flatten(),
                biases_color[0],
                ...
                biases_color[-1],
            ]
        )

    Here, `weights_XXX[i]` correspond to a `(M, N)` tensor storing the weight matrix
    of the i-th MLP layer. Similarly, `biases_XXX[i]` is a `(N,)` tensor storing
    the bias vector.

    The MLP multiplies the input features from the right, i.e.::

        output[i+1] = input[i] @ weights_XXX[i] + biases_XXX[i]

    Hence, `M` / `N` is the input / output channel dimension.

    In addition to the `mlp_params`, the `DecoderParams` class stores the number
    of hidden units each MLP. Specifically, `n_hidden_trunk`, `n_hidden_opacity`, and
    `n_hidden_color` are tensors of shape `(n_layers+1,)` that store the number of
    input channels followed by the output channel number of each layer in
    the trunk, opacity, and color MLPs, respectively.

    Note:
        One can convert the 1D `mlp_params` tensor to the more-interpretable
        list of weight matrices and bias tensors using the `flattened_decoder_params_to_list`
        function.

    Note:
        Since the Triton language of Lightplane's GPU kernel constraints the number
        of rendering channels to at least 16, the `color_chn` attribute is used to store
        the effective number of rendered output channels. If the effective number of
        rendered channels is less than 16, the MLP parameters are padded with zeros
        to match the minimum size.

    Attributes:
        mlp_params: The parameters for the Lightplane Rendering decoder.
        n_hidden_trunk: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `trunk_mlp`. Note that this tensor can be empty if the trunk MLP is not used.
        n_hidden_opacity: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `opacity_mlp`.
        n_hidden_color: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `color_mlp`.
        color_chn: The number of rendered channels.
    """
    mlp_params: torch.Tensor
    n_hidden_trunk: torch.Tensor
    n_hidden_opacity: torch.Tensor
    n_hidden_color: torch.Tensor
    color_chn: int


@dataclass
class SplatterParams:
    """
    Class representing learnable parameters of the MLP from Lightplane Splatter.

    The splatter comprises a learnable function that predicts a vector splatted
    to the output 3D feature grid. Specifically, the function is defined as follows::

        MLP(feature_grid[x] + splatting_feature[u]) -> splat_vector[x]

    where `x` corresponds to the 3D point along the the ray of pixel `u`,
    `feature_grid[x]` is the input shape grid sampled at point `x`, and
    `splatting_feature[u]` is the splatted feature at pixel `u`.
    The splatting MLP outputs `splat_vector[x]` which is pushed back into the
    output grid.

    The parameters of the MLP are stored in the `mlp_params` attribute.
    Here, `mlp_params` is a 1D tensor which concatenates the flattened weight matrices
    and bias vectors of the MLP in the following order::

        mlp_params = torch.cat(
            [
                weights[0].flatten(),
                ...
                weights[-1].flatten(),
                biases[0],
                ...
                biases[-1],
            ]
        )

    Here, `weights[i]` correspond to a `(M, N)` tensor storing the weight matrix
    of the i-th MLP layer. Similarly, `biases[i]` is a `(N,)` tensor storing
    the bias vector.

    The MLP multiplies the input features from the right, i.e.::

        output[i+1] = input[i] @ weights[i] + biases[i]

    Hence, `M` / `N` is the input / output channel dimension.

    In addition to the `mlp_params`, the `SplatterParams` class stores the number
    of MLP's hidden units. Specifically, the `n_hidden` field is a tensor of shape
    `(n_layers+1,)` that stores the number of input channels followed by
    the output channel number of each layer in the MLP.

    Attributes:
        mlp_params: The parameters for the Lightplane rendering decoder.
        n_hidden: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            splatting MLP.
    """

    mlp_params: torch.Tensor
    n_hidden: torch.Tensor


def init_decoder_params(
    device: torch.device,
    n_layers_opacity: int,
    n_layers_trunk: int,
    n_layers_color: int,
    input_chn: int = 32,
    hidden_chn: int = 32,
    color_chn: int = 3,
    opacity_init_bias: float = 0.0,
    pad_color_channels_to_min_block_size: bool = True,
    use_separate_color_grid: bool = False,
) -> DecoderParams:
    """
    The function initializes the learnable parameters of the Lightplane Renderer
    decoder given mlp configurations.
    Weights and biases of three MLPs inside decoder (`trunk_mlp`, `opacity_mlp`,
    and `color_mlp`) are initialized using Xavier initialization by function `_xavier_init_mlp_params`,
    and are flattened into a single tensor `mlp_params` by function `flatten_decoder_params`.

    Since the Triton language of Lightplane's GPU kernel constraints the number
    of rendering channels to at least 16, the `color_chn` attribute is used to store
    the effective number of rendered output channels. If the effective number of
    rendered channels is less than 16, the MLP parameters are padded with zeros
    to match the minimum size.

    Args:
        device: The device to store the parameters.
        n_hidden_trunk: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `trunk_mlp`. Note that this tensor can be empty if the trunk MLP is not used.
        n_hidden_opacity: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `opacity_mlp`.
        n_hidden_color: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `color_mlp`.
        input_chn: The number of input channels, which is the number of channel for
            `feature_grid`.
        hidden_chn: The number of hidden units in the MLP layers.
        color_chn: The number of rendered channels.
        opacity_init_bias: The initial bias value for the opacity MLP.
        pad_color_channels_to_min_block_size: If True, the MLP parameters are padded with zeros
            to match the minimum size of the triton minimum block size.
        use_separate_color_grid: If True, the renderer uses a separate color grid.
    """
    if n_layers_trunk > 0:
        assert not use_separate_color_grid, (
            "Cannot use trunk MLP with a separate color grid."
            " Please set n_layers_trunk==0."
        )
        (weights_trunk, biases_trunk,) = _xavier_init_mlp_params(
            n_layers_trunk,
            input_chn,
            hidden_chn,
            hidden_chn,
            device,
        )
    else:
        weights_trunk = []
        biases_trunk = []

    (weights_opacity, biases_opacity,) = _xavier_init_mlp_params(
        n_layers_opacity,
        input_chn if use_separate_color_grid else hidden_chn,
        hidden_chn,
        1,
        device,
        last_bias=opacity_init_bias,
    )

    (weights_color, biases_color,) = _xavier_init_mlp_params(
        n_layers_color,
        input_chn if use_separate_color_grid else hidden_chn,
        hidden_chn,
        color_chn,
        device,
    )

    # for p in [
    #     *weights_trunk, *biases_trunk,
    #     *weights_opacity, *biases_opacity,
    #     *weights_color, *biases_color,
    # ]:
    #     print(p.shape)

    # set the mlp params
    (
        mlp_params,
        n_hidden_trunk,
        n_hidden_opacity,
        n_hidden_color,
    ) = flatten_decoder_params(
        weights_trunk,
        biases_trunk,
        weights_opacity,
        biases_opacity,
        weights_color,
        biases_color,
        pad_color_channels_to_min_block_size,
    )

    return DecoderParams(
        mlp_params,
        n_hidden_trunk,
        n_hidden_opacity,
        n_hidden_color,
        color_chn,
    )


def init_splatter_params(
    device: torch.device,
    n_layers: int,
    input_chn: int = 32,
    hidden_chn: int = 32,
    out_chn: int = 16,
) -> SplatterParams:
    """
    The function initializes the learnable parameters of the Lightplane Splatter
    given mlp configurations.
    Weights and biases of the MLP inside LightPlane Splatter are initialized using
    Xavier initialization by function `_xavier_init_mlp_params`,
    and are flattened into a single tensor `mlp_params` by function `flatten_splatter_params`.

    Since the outout of the mlp is a vector splatted to the output 3D feature grid,
    whose number of channels is the same as the `output_grid`, which is typically more
    than 16.
    So we do not need to pad the MLP parameters to match the minimum size of the triton
    minimum block size.

    Args:
        device: The device to store the parameters.
        n_layers: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            mlp.
        input_chn: The number of input channels.
        hidden_chn: The number of hidden units in the MLP layers.
        out_chn: The number of output channels.
    """
    (weights, biases) = _xavier_init_mlp_params(
        n_layers, input_chn, hidden_chn, out_chn, device
    )

    mlp_params, n_hidden = flatten_splatter_params(
        weights,
        biases,
    )

    return SplatterParams(
        mlp_params,
        n_hidden,
    )


def get_triton_function_input_dims(
    n_hidden_trunk: torch.Tensor,
    n_hidden_opacity: torch.Tensor,
    n_hidden_color: torch.Tensor,
):
    """
    Function to get the MLP layers and hidden units from `n_hidden_trunk`,
    `n_hidden_opacity` and `n_hidden_color` inside `decoder_params` object.
    """
    if n_hidden_trunk.numel() == 0:
        # no trunk mlp
        mlp_n_layers_trunk = 0
        mlp_dim_hidden_trunk = 0
        mlp_dim_hidden_opacity, mlp_dim_hidden_color = (
            int(h[1].item()) for h in [n_hidden_opacity, n_hidden_color]
        )
    else:
        mlp_dim_hidden_trunk, mlp_dim_hidden_opacity, mlp_dim_hidden_color = (
            int(h[1].item()) for h in [n_hidden_trunk, n_hidden_opacity, n_hidden_color]
        )
        # all trunk hidden layers have to have the same number of hidden units
        assert (n_hidden_trunk[1:] == mlp_dim_hidden_trunk).all()
        mlp_n_layers_trunk = int(len(n_hidden_trunk)) - 1

    if n_hidden_opacity.numel() > 3:
        assert (mlp_dim_hidden_opacity == n_hidden_opacity[1:-1]).all()
    if n_hidden_color.numel() > 3:
        assert (mlp_dim_hidden_color == n_hidden_color[1:-1]).all()

    num_render_channels = n_hidden_color[-1].item()
    mlp_n_layers_opacity = int(len(n_hidden_opacity)) - 1
    mlp_n_layers_color = int(len(n_hidden_color)) - 1
    return (
        mlp_dim_hidden_trunk,
        mlp_dim_hidden_opacity,
        mlp_dim_hidden_color,
        mlp_n_layers_trunk,
        mlp_n_layers_opacity,
        mlp_n_layers_color,
        num_render_channels,
    )


# ------------------------
# --- Helper functions ---
# ------------------------


def flatten_decoder_params(
    weights_trunk: Tuple[torch.Tensor, ...],
    biases_trunk: Tuple[torch.Tensor, ...],
    weights_opacity: Tuple[torch.Tensor, ...],
    biases_opacity: Tuple[torch.Tensor, ...],
    weights_color: Tuple[torch.Tensor, ...],
    biases_color: Tuple[torch.Tensor, ...],
    pad_color_channels_to_min_block_size: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The hepler function to flatten the decoder parameters into a single tensor,
    and get the number of hidden units for each layer in each MLP (`n_hidden_XX`).

    Args:
        weights_trunk: Tuple of weight matrices for `trunk_mlp`.
        biases_trunk: Tuple of bias vectors for `trunk_mlp`.
        weights_opacity: Tuple of weight matrices for `opacity_mlp`.
        biases_opacity: Tuple of bias vectors for `opacity_mlp`.
        weights_color: Tuple of weight matrices for `color_mlp`.
        biases_color: Tuple of bias vectors for `color_mlp`.
        pad_color_channels_to_min_block_size: If True, the MLP parameters are padded with zeros
            to match the minimum size of the triton minimum block size.
    """
    # TODO: return the flattened param vector from DecoderParams directly
    num_pad_channels_color = 0
    if pad_color_channels_to_min_block_size:
        color_chn = biases_color[-1].numel()
        num_pad_channels_color = max(MIN_BLOCK_SIZE - color_chn, 0)

    if num_pad_channels_color > 0:
        weights_color, biases_color = _pad_color_mlp_params(
            weights_color,
            biases_color,
            num_pad_channels_color,
        )

    mlp_params = torch.cat(
        [
            t_elem.reshape(-1).contiguous()
            for t in [
                weights_trunk,
                biases_trunk,
                weights_opacity,
                biases_opacity,
                weights_color,
                biases_color,
            ]
            for t_elem in t
        ],
        dim=0,
    ).contiguous()

    # set the numbers of hidden units in each mlp
    n_hidden_trunk, n_hidden_opacity, n_hidden_color = (
        _get_n_hidden(w, device=mlp_params.device)
        for w in [weights_trunk, weights_opacity, weights_color]
    )

    _validate_flattened_mlp_params(
        mlp_params,
        n_hidden_trunk,
        n_hidden_opacity,
        n_hidden_color,
        pad_color_channels_to_min_block_size=pad_color_channels_to_min_block_size,
    )

    return mlp_params, n_hidden_trunk, n_hidden_opacity, n_hidden_color


def flatten_splatter_params(
    weights: Tuple[torch.Tensor, ...],
    biases: Tuple[torch.Tensor, ...],
):
    """
    The hepler function to flatten the splatter parameters into a single tensor,
    and get the number of hidden units for each layer in the MLP (`n_hidden`).

    Args:
        weights: Tuple of weight matrices for the MLP.
        biases: Tuple of bias vectors for the MLP.
    """
    mlp_params = torch.cat(
        [
            t_elem.reshape(-1).contiguous()
            for t in [
                weights,
                biases,
            ]
            for t_elem in t
        ],
        dim=0,
    ).contiguous()

    # set the numbers of hidden units in each mlp
    n_hidden = _get_n_hidden(weights, device=mlp_params.device)

    return mlp_params, n_hidden


def flattened_decoder_params_to_list(
    mlp_params: torch.Tensor,
    n_hidden_trunk: torch.Tensor,
    n_hidden_opacity: torch.Tensor,
    n_hidden_color: torch.Tensor,
    transpose: bool = False,
) -> Tuple[
    Tuple[torch.Tensor, ...],
    Tuple[torch.Tensor, ...],
    Tuple[torch.Tensor, ...],
    Tuple[torch.Tensor, ...],
    Tuple[torch.Tensor, ...],
    Tuple[torch.Tensor, ...],
]:
    """
    This function converts the flattened MLP parameters into a list of weight matrices,
    and bias vectors for each MLP.
    It is the inverse function of `flatten_decoder_params`.

    Args:
        mlp_params: The flattened MLP parameters, i.e. 1D tensor.
        n_hidden_trunk: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `trunk_mlp`. Note that this tensor can be empty if the trunk MLP is not used.
        n_hidden_opacity: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `opacity_mlp`.
        n_hidden_color: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `color_mlp`.
        transpose: If True, the weight matrices are transposed.
        
    Returns:
        weights_trunk: Weight matrices of the trunk MLP.
        biases_trunk: Bias vectors of the trunk MLP.
        weights_opacity: Weight matrices of the opacity MLP.
        biases_opacity: Bias vectors of the opacity MLP.
        weights_color: Weight matrices of the color MLP.
        biases_color: Bias vectors of the color MLP.
    """
    numel_trunk, numel_opacity, numel_color = (
        (nh[:-1].to(torch.float) @ nh[1:].to(torch.float)).to(torch.int32)
        + nh[1:].sum()
        for nh in [n_hidden_trunk, n_hidden_opacity, n_hidden_color]
    )

    weights_trunk, biases_trunk = _flattened_one_mlp_params_to_list(
        mlp_params[:numel_trunk],
        n_hidden_trunk,
        transpose,
    )

    weights_opacity, biases_opacity = _flattened_one_mlp_params_to_list(
        mlp_params[numel_trunk : (numel_trunk + numel_opacity)],
        n_hidden_opacity,
        transpose,
    )

    weights_color, biases_color = _flattened_one_mlp_params_to_list(
        mlp_params[(numel_trunk + numel_opacity) :],
        n_hidden_color,
        transpose,
    )

    return (
        weights_trunk,
        biases_trunk,
        weights_opacity,
        biases_opacity,
        weights_color,
        biases_color,
    )


def flattened_triton_decoder_to_list(
    mlp_params: torch.Tensor,
    n_layers_trunk: int,
    n_layers_opacity: int,
    n_layers_color: int,
    input_chn: int,
    hidden_chn: int,
    color_chn: int,
):
    """
    Another helper function to convert the flattened MLP parameters into a list
    of weight matrices, and bias vectors for each MLP.
    Given `mlp_params`, the number of layers for each MLP, input/output number
    of channesl, and hidden units number, this function returns the list of weight
    matrices and bias vectors for each MLP.

    Args:
        mlp_params: The flattened MLP parameters, i.e. 1D tensor.
        n_layers_trunk: The number of layers in the `trunk_mlp`.
        n_layers_opacity: The number of layers in the `opacity_mlp`.
        n_layers_color: The number of layers in the `color_mlp`.
        input_chn: The number of input channels.
        hidden_chn: The number of hidden units in the MLP layers.
        color_chn: The number of rendered channels.
    """

    def _make_n_hidden(dim_in, dim_hidden, dim_out, n_layers):
        n_hidden = [dim_in]
        for _ in range(n_layers - 1):
            n_hidden.append(dim_hidden)
        n_hidden.append(dim_out)
        return torch.tensor(n_hidden, dtype=torch.int32, device=mlp_params.device)

    n_hidden_trunk = _make_n_hidden(input_chn, hidden_chn, hidden_chn, n_layers_trunk)
    n_hidden_opacity = _make_n_hidden(hidden_chn, hidden_chn, 1, n_layers_opacity)
    n_hidden_color = _make_n_hidden(hidden_chn, hidden_chn, color_chn, n_layers_color)
    return flattened_decoder_params_to_list(
        mlp_params,
        n_hidden_trunk,
        n_hidden_opacity,
        n_hidden_color,
        transpose=False,
    )


# --------------------------------
# --- Helper private functions ---
# --------------------------------


def _validate_flattened_mlp_params(
    mlp_params: torch.Tensor,
    n_hidden_trunk: torch.Tensor,
    n_hidden_opacity: torch.Tensor,
    n_hidden_color: torch.Tensor,
    pad_color_channels_to_min_block_size: bool = False,
):
    """
    A helper function to validate whether the size of `mlp_params` satisfies the
    configuration specified by `n_hidden_trunk`, `n_hidden_opacity`, and
    `n_hidden_color`.

    Args:
        mlp_params: The flattened MLP parameters, i.e. 1D tensor.
        n_hidden_trunk: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `trunk_mlp`. Note that this tensor can be empty if the trunk MLP is not used.
        n_hidden_opacity: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `opacity_mlp`.
        n_hidden_color: `(n_layers+1,)` Long tensor storing the number of
            input channels followed by the number of hidden units in each layer of the
            `color_mlp`.
        pad_color_channels_to_min_block_size: If True, the MLP parameters are padded with zeros
            to match the minimum size of the triton minimum block size.
    """
    assert n_hidden_trunk.dtype == torch.int32
    assert n_hidden_opacity.dtype == torch.int32
    assert n_hidden_color.dtype == torch.int32
    assert mlp_params.dtype == torch.float

    (
        weights_trunk,
        biases_trunk,
        weights_opacity,
        biases_opacity,
        weights_color,
        biases_color,
    ) = flattened_decoder_params_to_list(
        mlp_params,
        n_hidden_trunk,
        n_hidden_opacity,
        n_hidden_color,
        transpose=False,
    )

    for w, b in (
        (weights_trunk, biases_trunk),
        (weights_opacity, biases_opacity),
        (weights_color, biases_color),
    ):
        _validate_mlp_params_list(w, b)

    if pad_color_channels_to_min_block_size:
        assert biases_color[-1].numel() >= MIN_BLOCK_SIZE


def _validate_mlp_params_list(
    weights_list: Tuple[torch.Tensor, ...],
    biases_list: Tuple[torch.Tensor, ...],
):
    """
    Helper function to validate the weight matrices and bias vectors of an MLP.
    It checks the shape and device of the weights and biases, and the consistency
    of the dimensions between the layers.
    """
    for l, (w, b) in enumerate(zip(weights_list, biases_list)):
        dim_in = w.shape[0]
        dim_out = w.shape[1]
        assert w.device == b.device
        assert b.ndim == 1
        assert w.ndim == 2
        assert dim_out == b.shape[0]
        if l > 0:
            w_prev = weights_list[l - 1]
            assert w_prev.shape[1] == dim_in


def _flattened_one_mlp_params_to_list(
    mlp_params: torch.Tensor,
    n_hidden: torch.Tensor,
    transpose: bool = False,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    """
    Helper function to convert the flattened MLP parameters into a list of weight matrices
    """
    nl = n_hidden.shape[0] - 1

    indims = n_hidden[:nl].tolist()
    outdims = n_hidden[1:].tolist()
    numels = n_hidden[:-1] * n_hidden[1:]
    tot_numel = numels.sum()
    w_mlp_params, b_mlp_params = mlp_params[:tot_numel], mlp_params[tot_numel:]
    assert w_mlp_params.numel() == tot_numel
    assert b_mlp_params.numel() == sum(outdims)
    weights = [
        w.reshape(indim, outdim)
        for w, indim, outdim in zip(
            w_mlp_params.split(numels.tolist()),
            indims,
            outdims,
        )
    ]
    biases = b_mlp_params.split(outdims)

    if transpose:
        weights = [w.t().contiguous() for w in weights]

    return weights, biases


def _get_n_hidden(
    w: Tuple[torch.Tensor, ...],
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to get the number of hidden units for each layer in the MLP.
    """
    if len(w) == 0:
        return torch.tensor([], dtype=torch.int32, device=device)
    n_hidden = [w_.shape[1] for w_ in w]
    n_hidden.insert(0, w[0].shape[0])
    n_hidden = torch.tensor(
        n_hidden,
        dtype=torch.int32,
        device=device,
    ).contiguous()
    return n_hidden


def _pad_color_mlp_params(
    weights: Tuple[torch.Tensor],
    biases: Tuple[torch.Tensor],
    n_pad: int,
):
    """
    Helper function to pad the MLP parameters with zeros to match the minimum output
    size.
    """
    weights[-1] = torch.nn.functional.pad(weights[-1], [0, n_pad])
    biases[-1] = torch.nn.functional.pad(biases[-1], [0, n_pad])
    return weights, biases


def _xavier_init_mlp_params(
    n_layers: int,
    input_chn: int,
    hidden_chn: int,
    output_chn: int,
    device: torch.device,
    last_bias: float = 0.0,
    last_num_pad_channels: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to initialize the weights and biases of an MLP with Xavier
    initialization and zero padding for output channels.
    """
    weights = [
        torch.empty(
            input_chn if l == 0 else hidden_chn,
            output_chn if l == n_layers - 1 else hidden_chn,
            device=device,
        )
        for l in range(n_layers)
    ]
    for wi, w in enumerate(weights):  # xavier init the weights
        w_init = w
        torch.nn.init.xavier_uniform_(w_init, gain=torch.nn.init.calculate_gain("relu"))
        weights[wi] = w_init.contiguous()

    biases = [
        (
            torch.full((output_chn,), device=device, fill_value=last_bias)
            if l == n_layers - 1
            else torch.zeros(hidden_chn, device=device)
        )
        for l in range(n_layers)
    ]

    if last_num_pad_channels > 0:
        weights[-1] = torch.cat(
            [
                weights[-1],
                torch.zeros(
                    output_chn,
                    last_num_pad_channels,
                    device=device,
                ),
            ],
            dim=1,
        )
        biases[-1] = torch.cat(
            [
                biases[-1],
                torch.zeros(
                    last_num_pad_channels,
                    device=device,
                ),
            ],
            dim=0,
        )

    return weights, biases
