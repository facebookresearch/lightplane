# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def get_generated_file_name(
    template_name, N_LAYERS_TRUNK, N_LAYERS_OPACITY, N_LAYERS_COLOR
):
    return f"{template_name}_t{N_LAYERS_TRUNK}_o{N_LAYERS_OPACITY}_c{N_LAYERS_COLOR}"


def get_generated_splatter_file_name(template_name, N_LAYERS):
    return f"{template_name}_mlp{N_LAYERS}"


def get_enumerated_str(n, prefix, postfix):
    return ", ".join(f"{prefix}{i}{postfix}" for i in range(n))


def get_wb_str(mlp_name, n_layers):
    if n_layers == 0:
        return ""
    postfix = f"_{mlp_name}" if mlp_name is not None else ""
    w_str = get_enumerated_str(n_layers, "w", postfix)
    b_str = get_enumerated_str(n_layers, "b", postfix)
    wb_str = w_str + ", " + b_str
    return wb_str


def get_dwb_str(mlp_name, n_layers):
    if n_layers == 0:
        return ""
    postfix = f"_{mlp_name}" if mlp_name is not None else ""
    w_str = get_enumerated_str(n_layers, "dw", postfix)
    b_str = get_enumerated_str(n_layers, "db", postfix)
    wb_str = w_str + ", " + b_str
    return wb_str


def get_xwb_str(mlp_name, n_layers):
    if n_layers == 0:
        return ""
    postfix = f"_{mlp_name}" if mlp_name is not None else ""
    xwb_str = []
    for l in range(int(n_layers)):
        if (l < int(n_layers) - 1) or (mlp_name == "trunk") or (mlp_name == "TRUNK"):
            xwb_str.append(f"xwb{l}{postfix}")
    if len(xwb_str) == 0:
        return None
    return ", ".join(xwb_str)


def get_x_str(mlp_name, n_layers):
    if n_layers == 0:
        return ""
    postfix = f"_{mlp_name}" if mlp_name is not None else ""
    x_str = get_enumerated_str(n_layers, "x", postfix)
    return x_str


def get_zerowb_str(mlp_name, n_layers):
    if n_layers == 0:
        return ""
    postfix = f"_{mlp_name}" if mlp_name is not None else ""
    w_str = get_enumerated_str(n_layers, "zero_w", postfix)
    b_str = get_enumerated_str(n_layers, "zero_b", postfix)
    wb_str = w_str + ", " + b_str
    return wb_str
