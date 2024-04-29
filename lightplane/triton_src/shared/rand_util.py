# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
import triton
import triton.language as tl


INT32_PRIME = 105097564  # the largest int32 prime
MAX_INT_32_F = 2147483647.0
MAX_UINT_32_F = 4294967295.0
MAX_UINT_32_F_EPS = 3.0


@triton.jit
def int_to_randn_kernel(
    x1,
    x2,
    out,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    seed: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_mask = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < N
    x1_buffer = tl.load(x1 + offs, mask=offs_mask).to(tl.int32)
    x2_buffer = tl.load(x2 + offs, mask=offs_mask).to(tl.int32)
    seed_buffer = tl.full((BLOCK_SIZE,), seed, dtype=tl.int64).to(tl.int32)
    r = int_to_randn(x1_buffer, x2_buffer, seed_buffer)
    tl.store(out + offs, r, mask=offs_mask)


@triton.jit
def hash(x):  # x is tl.int32
    # https://stackoverflow.com/a/12996028
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = (x >> 16) ^ x
    return x


@triton.jit
def pair_hash(x, h):  # x, h is tl.int32
    # https://stackoverflow.com/a/30057527
    h = h ^ x
    h = (h << 24) + h * 0x193
    return h


@triton.jit
def int32_to_float01(x):  # x is tl.int32
    x_01 = (x.to(tl.float32) + MAX_INT_32_F + MAX_UINT_32_F_EPS) / (
        MAX_UINT_32_F + MAX_UINT_32_F_EPS
    )
    return x_01


@triton.jit
def int_to_randn(x1, x2, seed):  # x is tl.uint32
    # convert two integers to a float which is randomly-normally distributed
    # 1) hash both ints to a uniformly distributed uint32
    # 2) divide by max uint32 to map to [0, 1] (uniform distribution due to hashing fun)
    # 3) box-muller transform to map U[0, 1] to N(0, 1)
    x_hash_1 = hash(x1.to(tl.int32))
    x_hash_2 = hash(x2.to(tl.int32))
    # alter the hashes with the seed: https://stackoverflow.com/a/30057527
    x_hash_1 = pair_hash(pair_hash(INT32_PRIME, seed), x_hash_1)  # slower+stronger
    x_hash_2 = pair_hash(pair_hash(INT32_PRIME, seed + 1), x_hash_2)
    # transform to [0, 1]
    x_01_1 = int32_to_float01(x_hash_1)
    x_01_2 = int32_to_float01(x_hash_2)
    # box-muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    z = tl.sqrt(-2 * tl.log(x_01_1)) * tl.cos(6.28318530718 * x_01_2)
    return z


@triton.jit
def get_sample_randn(pid, step, n_rays, n_steps, BLOCK_SIZE, seed_buffer):
    offs = pid * BLOCK_SIZE * n_steps + 1
    i1 = offs + step + tl.arange(0, BLOCK_SIZE) * n_steps
    i2 = n_rays * n_steps + i1
    return int_to_randn(i1, i2, seed_buffer)


def int_to_randn_triton(x1, x2, seed: int, BLOCK_SIZE: int = 256):
    N = x1.numel()
    z = x1.new_empty(N).float()
    n_blocks = int(math.ceil(N / BLOCK_SIZE))
    int_to_randn_kernel[(n_blocks,)](
        x1,
        x2,
        z,
        N,
        BLOCK_SIZE,
        seed,
    )
    return z


# -------
# PyTorch implementation
# -------


def _hash_naive(x):
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = (x >> 16) ^ x
    return x


def _pair_hash_naive(x, h):  # x, h is tl.uint32
    # https://stackoverflow.com/a/30057527
    h = h ^ x
    h = (h << 24) + h * 0x193
    return h


def int_to_randn_naive(
    x1,
    x2,
    seed: int,
):
    x1, x2 = [x.int() for x in [x1, x2]]
    x_hash_1 = _hash_naive(x1.int())
    x_hash_2 = _hash_naive(x2.int())
    x_hash_1 = _pair_hash_naive(
        _pair_hash_naive(INT32_PRIME, seed), x_hash_1
    )  # slower+stronger
    x_hash_2 = _pair_hash_naive(_pair_hash_naive(INT32_PRIME, seed + 1), x_hash_2)
    # transform to [0, 1]
    x_hash_1_f = x_hash_1.int() + MAX_INT_32_F
    x_hash_2_f = x_hash_2.int() + MAX_INT_32_F
    x_01_1 = (x_hash_1_f + MAX_UINT_32_F_EPS) / (
        MAX_UINT_32_F + MAX_UINT_32_F_EPS
    )  # 4294967295.0 = max uint32
    x_01_2 = (x_hash_2_f + MAX_UINT_32_F_EPS) / (MAX_UINT_32_F + MAX_UINT_32_F_EPS)
    # box-muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    z = (-2 * x_01_1.log()).sqrt() * (6.28318530718 * x_01_2).cos()
    return z
