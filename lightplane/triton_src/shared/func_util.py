# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import triton
import triton.language as tl
from .const import ALLOW_TF32, ALLOW_WARP_DIVERGENCE


@triton.jit
def d_sigmoid(dy, x):
    s = tl.sigmoid(x)
    return dy * s * (1 - s)


@triton.jit
def _softplus(x):
    z = tl.where(x >= 0, x + tl.log(1 + tl.exp(-x)), tl.log(1 + tl.exp(x)))
    return z


@triton.jit
def _d_softplus(grad, x):
    z = tl.where(x >= 0, 1 / (1 + tl.exp(-x)), 1 - 1 / (1 + tl.exp(x)))
    return grad * z


if ALLOW_WARP_DIVERGENCE:
    # no tl.debug_barrier() calls -> can lead to warp divergence and slow-downs
    @triton.jit
    def softplus(x):
        return _softplus(x)

    @triton.jit
    def d_softplus(grad, x):
        return _d_softplus(grad, x)

else:
    # calls tl.debug_barrier() to avoid warp divergence
    @triton.jit
    def softplus(x):
        z = _softplus(x)
        tl.debug_barrier()
        return z

    @triton.jit
    def d_softplus(grad, x):
        z = _d_softplus(grad, x)
        tl.debug_barrier()
        return z


@triton.jit
def d_linear_relu(d_y, w, b, xwb, x):
    # gradients of `y = max(x @ w + b, 0); xwb = x @ w + b`
    d_y_relu = d_y * (xwb > 0.0).to(tl.float32)
    return d_linear(d_y_relu, w, b, x)


@triton.jit
def d_linear(d_y, w, b, x):
    # gradients of `y = x @ w + b;
    d_x = tl.dot(d_y, tl.trans(w), allow_tf32=ALLOW_TF32)
    d_w = tl.dot(tl.trans(d_y), x, allow_tf32=ALLOW_TF32)
    d_b = tl.sum(d_y, axis=0)
    return d_x, d_w, d_b
