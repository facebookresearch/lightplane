# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import triton
import triton.language as tl


@triton.jit
def contract_pi(x, y, z):
    # MERF contract_pi function
    # def contract_pi(x):
    #     n = x.abs().max(dim=-1).values[..., None]
    #     x_contract = torch.where(
    #         n <= 1.0,
    #         x,
    #         torch.where(
    #             (x.abs()-n).abs() <= 1e-7,
    #             (2 - 1/x.abs()) * (x / x.abs()),
    #             x / n,
    #         )
    #     )
    #     return x_contract
    n = tl.maximum(tl.maximum(tl.abs(x), tl.abs(y)), tl.abs(z))
    x_c = _contract_pi_one(x, n)
    y_c = _contract_pi_one(y, n)
    z_c = _contract_pi_one(z, n)
    return x_c, y_c, z_c


@triton.jit
def _contract_pi_one(x, n):
    x_c = tl.where(
        n <= 1.0,
        x,
        tl.where(
            tl.abs(tl.abs(x) - n) <= 1e-8, (2 - 1 / tl.abs(x)) * (x / tl.abs(x)), x / n
        ),
    )
    # important: we map the contracted coords from [-2, 2] to [-1, 1]!
    x_c = x_c * 0.5
    return x_c


@triton.jit
def depth_inv_sphere(far, disparity_at_inf, n, step):
    frac_step = (step + 1) / n
    n_disp = (disparity_at_inf - 1) * frac_step + 1
    return far * (1 / n_disp)


@triton.jit
def depth_lin(near, far, n, step):
    frac_step = step / (n - 1)
    return (far - near) * frac_step + near
