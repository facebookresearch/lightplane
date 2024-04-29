# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# [[[cog
#
# # example call: > cog -d -D N_LAYERS=2 COG_UTIL_MODULE=lightplane.triton_src.templates.cog_util MLP_UTIL_MODULE=lightplane.triton_src.generated.mlp_util_2 ./mlp_util.py
# # This `cog` template expects the following constants:
# # N_LAYERS: number of mlp layers
# # COG_UTIL_MODULE: the module name of the cog_util.py file
# # MLP_UTIL_MODULE: the module name of the mlp_util.py file
#
# import cog
# import importlib
# N_LAYERS = int(N_LAYERS)
# cog_util = importlib.import_module(COG_UTIL_MODULE)
# cog.outl("import triton")
# cog.outl("import triton.language as tl")
# cog.outl("from ..shared.const import ALLOW_TF32")
# def create_load_mlp_params_def(n_layers):
#     prev_offs = "0"
#     cog.outl("@triton.jit")
#     cog.outl(f"def load_mlp_params(mlp_params, DIM_IN, DIM_HIDDEN, DIM_OUT, BLOCK_SIZE):")
#     for load_weight in [True, False]:
#         dim_in = "DIM_IN"
#         for l in range(int(n_layers)):  # load weights
#             dim_out = f"DIM_OUT" if (l == int(n_layers) - 1) else f"DIM_HIDDEN"
#             if load_weight:
#                 cog.outl(f"    w{l}_offs = {prev_offs}")
#                 cog.outl(f"    w{l}_numel = {dim_in} * {dim_out}")
#                 cog.outl(f"    w{l} = load_weight(mlp_params + w{l}_offs, {dim_in}, {dim_out})")
#                 prev_offs = f"w{l}_offs + w{l}_numel"
#             else:
#                 cog.outl(f"    b{l}_offs = {prev_offs}")
#                 cog.outl(f"    b{l}_numel = {dim_out}")
#                 cog.outl(f"    b{l} = load_bias(mlp_params + b{l}_offs, {dim_out}, BLOCK_SIZE)")
#                 prev_offs = f"b{l}_offs + b{l}_numel"
#             dim_in = dim_out
#     if N_LAYERS > 0:
#       cog.outl(f"    return " +  cog_util.get_wb_str(None, n_layers))
#     else:
#       cog.outl(f"    return None")
#     cog.outl("")
#
# # help function to calculate gradients for MLPs
# cog.outl("@triton.jit")
# cog.outl("def _d_linear(d_y, w, b, x):")
# cog.outl("    # gradients of `y = x @ w + b")
# cog.outl("    d_x = tl.dot(d_y, tl.trans(w), allow_tf32=ALLOW_TF32)")
# cog.outl("    d_w = tl.trans(tl.dot(tl.trans(d_y), x, allow_tf32=ALLOW_TF32))")
# cog.outl("    d_b = tl.sum(d_y, axis=0)")
# cog.outl("    return d_x, d_w, d_b")
# cog.outl("")
# cog.outl("@triton.jit")
# cog.outl("def _d_linear_relu(d_y, w, b, xwb, x):")
# cog.outl("    # gradients of `y = max(x @ w + b, 0); xwb = x @ w + b`")
# cog.outl("    d_y_relu = d_y * (xwb > 0.0).to(tl.float32)")
# cog.outl("    return _d_linear(d_y_relu, w, b, x)")
# cog.outl("")
# cog.outl("@triton.jit")
# cog.outl(f"def load_weight(ptr, dim_in, dim_out):")
# cog.outl(f"   return load_weight_2dim(ptr, dim_in, dim_out)")
# cog.outl("")
# cog.outl("@triton.jit")
# cog.outl(f"def load_weight_2dim(ptr, dim_in, dim_out):")
# cog.outl(f"   offs = tl.arange(0, dim_in)[:, None] * dim_out + tl.arange(0, dim_out)[None, :]")
# cog.outl(f"   w = tl.view(tl.load(ptr + offs), (dim_in, dim_out))")
# cog.outl(f"   return w")
# cog.outl("")
# cog.outl("@triton.jit")
# cog.outl("def load_bias(ptr, dim, BLOCK_SIZE):")
# cog.outl("    return tl.view(tl.load((ptr + tl.arange(0, dim))[None, :] + tl.zeros((BLOCK_SIZE, 1), dtype=tl.int32)), (BLOCK_SIZE, dim))")
# cog.outl("")
# cog.outl("@triton.jit")
# cog.outl(f"def update_weight(ptr, dim_in, dim_out, grad):")
# cog.outl(f"    offs = tl.arange(0, dim_in)[:, None] * dim_out + tl.arange(0, dim_out)[None, :]")
# cog.outl(f"    tl.atomic_add(ptr + offs, grad)")
# cog.outl("")
# cog.outl("@triton.jit")
# cog.outl(f"def update_bias(ptr, dim, grad):")
# cog.outl(f"    offs = tl.arange(0, dim)")
# cog.outl(f"    tl.atomic_add(ptr + offs, grad)")
# cog.outl("")
# def create_mlp_def(n_layers):
#     wb_str = cog_util.get_wb_str(None, n_layers)
#     cog.outl("@triton.jit")
#     if N_LAYERS > 0:
#       cog.outl(f"def mlp_splatter(x, " + wb_str + "):")
#     else:
#       cog.outl(f"def mlp_splatter(x):")
#     for l in range(int(n_layers)):
#         cog.outl(f"    x = tl.dot(x, w{l}, allow_tf32=ALLOW_TF32) + b{l}")
#         if (l < int(n_layers) - 1):
#             cog.outl(f"    x = tl.maximum(x, 0.0)")
#     if N_LAYERS > 0:
#       cog.outl(f"    return x")
#     else:
#       cog.outl(f"    pass")
#     cog.outl("")
#
# def create_mlp_with_inter_feat_def(n_layers):
#     wb_str = cog_util.get_wb_str(None, n_layers)
#     cog.outl("@triton.jit")
#     if N_LAYERS > 0:
#       cog.outl(f"def mlp_splatter_with_inter_feat(x, " + wb_str + "):")
#     else:
#       cog.outl(f"def mlp_splatter_with_inter_feat(x):")
#     inter_x = ""
#     inter_xwb = ""
#     for l in range(int(n_layers)):
#         cog.outl(f"    x{l} = x")
#         inter_x = inter_x + f", x{l}"
#         cog.outl(f"    x = tl.dot(x, w{l}, allow_tf32=ALLOW_TF32) + b{l}")
#         if (l < int(n_layers) - 1):
#             cog.outl(f"    xwb{l} = x")
#             inter_xwb = inter_xwb + f", xwb{l}"
#             cog.outl(f"    x = tl.maximum(x, 0.0)")
#     if N_LAYERS > 0:
#       cog.outl(f"    return x{inter_x}{inter_xwb}")
#     else:
#       cog.outl(f"    pass")
#     cog.outl("")
#
#
# def create_update_mlp_params_def(n_layers):
#     prev_offs = "0"
#     cog.outl("@triton.jit")
#     if N_LAYERS > 0:
#       cog.outl(f"def update_mlp_params(mlp_params, DIM_IN, DIM_HIDDEN, DIM_OUT, {cog_util.get_dwb_str(None, n_layers)}):")
#     else:
#       cog.outl(f"def update_mlp_params(mlp_params, DIM_IN, DIM_HIDDEN, DIM_OUT):")
#     for load_weight in [True, False]:
#         dim_in = "DIM_IN"
#         for l in range(int(n_layers)):  # load weights
#             dim_out = f"DIM_OUT" if (l == int(n_layers) - 1) else f"DIM_HIDDEN"
#             if load_weight:
#                 cog.outl(f"    w{l}_offs = {prev_offs}")
#                 cog.outl(f"    w{l}_numel = {dim_in} * {dim_out}")
#                 cog.outl(f"    update_weight(mlp_params + w{l}_offs, {dim_in}, {dim_out}, dw{l})")
#                 prev_offs = f"w{l}_offs + w{l}_numel"
#             else:
#                 cog.outl(f"    b{l}_offs = {prev_offs}")
#                 cog.outl(f"    b{l}_numel = {dim_out}")
#                 cog.outl(f"    update_bias(mlp_params + b{l}_offs, {dim_out}, db{l})")
#                 prev_offs = f"b{l}_offs + b{l}_numel"
#             dim_in = dim_out
#     if N_LAYERS > 0:
#       cog.outl("")
#     else:
#       cog.outl(f"    pass")
#     cog.outl("")
# def create_init_grad_def(n_layers):
#   dim_in = f"DIM_IN"
#   for l in range(n_layers):
#       dim_out = f"DIM_OUT" if (l == int(n_layers) - 1) else f"DIM_HIDDEN"
#       if l==int(n_layers)-1:
#           cog.outl(f"    dw{l} = tl.zeros((1, {dim_in}), dtype=tl.float32)")
#           cog.outl(f"    db{l} = tl.zeros(({dim_out},), dtype=tl.float32)")
#           cog.outl(f"    zero_w{l} = tl.zeros((1, {dim_in}), dtype=tl.float32)")
#           cog.outl(f"    zero_b{l} = tl.zeros(({dim_out},), dtype=tl.float32)")
#       else:
#           cog.outl(f"    dw{l} = tl.zeros(({dim_in}, {dim_out}), dtype=tl.float32)")
#           cog.outl(f"    db{l} = tl.zeros(({dim_out},), dtype=tl.float32)")
#           cog.outl(f"    zero_w{l} = tl.zeros(({dim_in}, {dim_out}), dtype=tl.float32)")
#           cog.outl(f"    zero_b{l} = tl.zeros(({dim_out},), dtype=tl.float32)")
#       dim_in = dim_out
#       cog.outl(" ")
#
# def create_d_mlp_def(n_layers):
#     mlp_name = "mlp"
#     wb_str = cog_util.get_wb_str(mlp_name, n_layers)
#     xwb_str = cog_util.get_xwb_str(mlp_name, n_layers)
#     x_str = cog_util.get_x_str(mlp_name, n_layers)
#     dwb_str = cog_util.get_dwb_str(mlp_name, n_layers)
#     cog.outl("@triton.jit")
#     if n_layers <= 0:
#         cog.outl("def d_mlp_splatter(dy):")
#         cog.outl("    return dy")
#         return None
#     if xwb_str is not None:
#         cog.outl(f"def d_mlp_splatter(dy, "+ wb_str + ", "+ xwb_str + ", "+ x_str + "):")
#     else:
#         cog.outl(f"def d_mlp_{mlp_name}(dy, "+ wb_str + ", "+ x_str + "):")
#     for l in reversed(range(int(n_layers))):
#         if (l < int(n_layers) - 1):
#             cog.outl(f"    dy, dw{l}_{mlp_name}, db{l}_{mlp_name} = _d_linear_relu(dy, w{l}_{mlp_name}, b{l}_{mlp_name}, xwb{l}_{mlp_name}, x{l}_{mlp_name})")
#         else:
#            cog.outl(f"    dy, dw{l}_{mlp_name}, db{l}_{mlp_name} = _d_linear(dy, w{l}_{mlp_name}, b{l}_{mlp_name}, x{l}_{mlp_name})")
#     cog.outl(f"    return dy, "+ dwb_str + "")
#     cog.outl("")
#
# def create_init_grad_def(n_layers):
#   mlp_name = "mlp"
#   dim_in = f"DIM_IN_{mlp_name}"
#   for l in range(n_layers):
#       dim_out = f"DIM_OUT_{mlp_name}" if (l == int(n_layers) - 1) else f"DIM_HIDDEN_{mlp_name}"
#       cog.outl(f"    dw{l}_{mlp_name} = tl.zeros(({dim_in}, {dim_out}), dtype=tl.float32)")
#       cog.outl(f"    db{l}_{mlp_name} = tl.zeros(({dim_out},), dtype=tl.float32)")
#       dim_in = dim_out
#       cog.outl(" ")
# create_load_mlp_params_def(N_LAYERS)
# create_mlp_def(N_LAYERS)
# cog.outl("# We re-define another function for feedforward which stores intermediate results for backpropogation")
# create_mlp_with_inter_feat_def(N_LAYERS)
# create_update_mlp_params_def(N_LAYERS)
# create_d_mlp_def(N_LAYERS)
# cog.outl("")
# cog.outl("# the main function for initialization grad and zero_grad buffer")
# cog.outl("@triton.jit")
# cog.outl("def init_mlp_params_grads(")
# for var_name in ("DIM_HIDDEN", "DIM_IN", "DIM_OUT"):
#     cog.outl(f"    {var_name}_mlp,")
# cog.outl("):")
#
# if N_LAYERS > 0:
#   create_init_grad_def(N_LAYERS)
#   cog.outl("    return(")
#   value_str = cog_util.get_dwb_str("mlp", N_LAYERS)
#   cog.outl(f"     {value_str},")
#   cog.outl("    )")
# else:
#   cog.outl("  return None")
# ]]]
# [[[end]]]
