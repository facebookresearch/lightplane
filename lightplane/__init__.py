# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from .lightplane_renderer import lightplane_renderer
from .lightplane_splatter import lightplane_mlp_splatter, lightplane_splatter
from .mlp_utils import (
    DecoderParams,
    SplatterParams,
    init_decoder_params,
    init_splatter_params,
    flatten_decoder_params,
    flatten_splatter_params,
    flattened_decoder_params_to_list,
    flattened_triton_decoder_to_list,
    get_triton_function_input_dims,
)
from .naive_renderer import lightplane_renderer_naive
from .naive_splatter import lightplane_splatter_naive, lightplane_mlp_splatter_naive
from .ray_utils import (
    Rays,
    calc_harmonic_embedding,
    calc_harmonic_embedding_dim,
    jitter_near_far,
)
from .renderer_module import LightplaneRenderer
from .splatter_module import LightplaneMLPSplatter, LightplaneSplatter
from .visualize import visualize_rays_plotly
