# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import unittest

import torch

import lightplane
from lightplane.triton_src.shared.rand_util import (
    int_to_randn_triton,
    int_to_randn_naive,
)

sys.path.insert(0, os.path.join(*lightplane.__file__.rsplit(os.sep, 2)[:-2], "tests"))
from utils import (
    seed_random_generators,
)


class TestRandn(unittest.TestCase):
    def setUp(self):
        seed_random_generators(0)

    def test(self):
        device = torch.device("cuda:0")
        N = 100000
        n_it = 10
        means = []
        stds = []
        for it in range(n_it):
            x1 = torch.randperm(N, device=device)
            x2 = torch.randperm(N, device=device)
            z = int_to_randn_naive(x1, x2, it)
            z_k = int_to_randn_triton(x1, x2, it)
            means.append(z_k.mean())
            stds.append(z_k.std())
            assert (z - z_k).abs().max() <= 1e-3
        assert torch.tensor(means).mean().abs() <= 0.01
        assert (torch.tensor(stds).mean() - 1).abs() <= 0.01


if __name__ == "__main__":
    unittest.main()
