# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import dataclasses
import itertools
import os
import sys
import unittest

import torch
from tqdm import tqdm

import lightplane

sys.path.insert(0, os.path.join(*lightplane.__file__.rsplit(os.sep, 2)[:-2], "tests"))
from utils import compare_one, get_test_example, seed_random_generators

from lightplane import lightplane_renderer, lightplane_renderer_naive

# If True, produces a verbose stdout-output, and stops unittest after the first failure.
DEBUG = True


class TestRendererWithAutograd(unittest.TestCase):
    def setUp(self, verbose: bool = DEBUG):
        seed_random_generators(0)
        self.verbose = verbose

    def test(self):
        test_sweep = {
            "grid_size_color": [None, [4, 3, 9]],
            "mask_out_of_bounds_samples": [True, False],
            "contract_coords": [False, True],
            "grid_size": [[3, 16, 12, 8, 16]],
            "one_ray": [False],
            "n_rays": [128, 3],
            "n_hidden": [32],
            "num_samples": [16],
            "num_samples_inf": [11, 0],
            "gain": [1.0, 3.0],
            "color_chn": [3],
            "scaffold_size": [[6, 4, 5], None],
            "is_triplane": [False, True],
            "n_layers_trunk": [2, 4],
            "n_layers_opacity": [2, 4],
            "n_layers_color": [2, 4],
            "inject_noise_sigma": [1.0, 0.0],
            "use_tensor_grid": [False, True]
            # TODO: do a sweep focused on n_layers
            # TODO: do a separate sweep checking n_rays < 16
        }

        test_sweep_nums = {k: list(range(len(v))) for k, v in test_sweep.items()}

        for p, np in zip(
            tqdm(list(itertools.product(*list(test_sweep.values())))),
            list(itertools.product(*list(test_sweep_nums.values()))),
        ):
            for it in range(5):
                p_dict = copy.deepcopy(dict(zip(test_sweep.keys(), p)))
                num_p_dict = copy.deepcopy(dict(zip(test_sweep.keys(), np)))

                use_separate_color_grid = p_dict["grid_size_color"] is not None
                if use_separate_color_grid:
                    p_dict["n_layers_trunk"] = 0

                test_example_params = {
                    k: p_dict[k]
                    for k in [
                        "grid_size",
                        "grid_size_color",
                        "n_rays",
                        "n_hidden",
                        "color_chn",
                        "n_layers_trunk",
                        "n_layers_opacity",
                        "n_layers_color",
                        "scaffold_size",
                        "one_ray",
                        "is_triplane",
                        "use_tensor_grid",
                    ]
                }

                with torch.random.fork_rng():
                    seed_random_generators(it)
                    (
                        rays,
                        grid,
                        grid_sizes,
                        color_grid,
                        color_grid_sizes,
                        mlp_params,
                        scaffold,
                    ) = get_test_example(
                        **test_example_params,
                        requires_grad=False,
                    )

                test_params_str_lines = [f"  {str('it'):30s}: {it}"]
                for k, v in p_dict.items():
                    test_params_str_lines.append(
                        f"  {k:30s}: {str(v):30s} ({num_p_dict[k] + 1} / {len(test_sweep[k])})"
                    )
                test_params_str = "\n" + "\n".join(test_params_str_lines)

                if self.verbose:
                    print("test params:")
                    print(test_params_str)

                with self.subTest(msg=test_params_str):
                    lighplane_args = [rays, grid, mlp_params]
                    lighplane_kwargs = dict(
                        color_grid=color_grid,
                        scaffold=scaffold,
                        num_samples=p_dict["num_samples"],
                        num_samples_inf=p_dict["num_samples_inf"],
                        gain=p_dict["gain"],
                        mask_out_of_bounds_samples=p_dict["mask_out_of_bounds_samples"],
                        inject_noise_sigma=p_dict["inject_noise_sigma"],
                        contract_coords=p_dict["contract_coords"],
                        inject_noise_seed=0,
                        disparity_at_inf=0.01,
                        grid_sizes=grid_sizes,
                        color_grid_sizes=color_grid_sizes,
                        triton_block_size=16,
                        triton_num_warps=4,
                    )
                    d_, m_, f_ = lightplane_renderer_naive(
                        *lighplane_args, **lighplane_kwargs
                    )
                    d, m, f = lightplane_renderer(*lighplane_args, **lighplane_kwargs)
                    if self.verbose:
                        print(f" forward check:")
                    test_results = [
                        compare_one(
                            d,
                            d_,
                            "  expected_depth",
                            run_assert=False,
                            verbose=self.verbose,
                        ),
                        compare_one(
                            m,
                            m_,
                            "  negative_log_trans",
                            run_assert=False,
                            verbose=self.verbose,
                        ),
                        compare_one(
                            f,
                            f_,
                            "  expected_feature",
                            run_assert=False,
                            verbose=self.verbose,
                        ),
                    ]

                    if not all(test_results):
                        assert False

                    grads = {"naive": {}, "lightplane": {}}
                    for is_naive in [False, True]:
                        with torch.random.fork_rng():
                            seed_random_generators(it)
                            (
                                rays,
                                grid,
                                grid_sizes,
                                color_grid,
                                color_grid_sizes,
                                mlp_params,
                                scaffold,
                            ) = get_test_example(
                                **test_example_params,
                                requires_grad=True,
                            )
                        lighplane_args = [rays, grid, mlp_params]
                        lighplane_kwargs = dict(
                            color_grid=color_grid,
                            scaffold=scaffold,
                            num_samples=p_dict["num_samples"],
                            num_samples_inf=p_dict["num_samples_inf"],
                            gain=p_dict["gain"],
                            mask_out_of_bounds_samples=p_dict[
                                "mask_out_of_bounds_samples"
                            ],
                            inject_noise_sigma=p_dict["inject_noise_sigma"],
                            inject_noise_seed=0,
                            disparity_at_inf=0.01,
                            grid_sizes=grid_sizes,
                            color_grid_sizes=color_grid_sizes,
                            triton_block_size=16,
                            triton_num_warps=4,
                        )

                        if is_naive:
                            fun = lightplane_renderer_naive
                        else:
                            fun = lightplane_renderer

                        d, m, f = fun(*lighplane_args, **lighplane_kwargs)

                        with torch.random.fork_rng():
                            seed_random_generators(it)
                            loss = sum(
                                (torch.randn_like(v) * v).sum() for v in [d, m, f]
                            )

                        loss.backward()

                        grad_tensors = {
                            **{
                                k: v.grad
                                for k, v in dataclasses.asdict(mlp_params).items()
                                if (
                                    (torch.is_tensor(v) and v.requires_grad)
                                    or (
                                        isinstance(v, list)
                                        and any(vv.requires_grad for vv in v)
                                    )
                                )
                            },
                            "enc": rays.encoding.grad,
                        }
                        if isinstance(grid, list):
                            grad_tensors.update(
                                {
                                    "grid": torch.cat(
                                        [
                                            g.grad.reshape(
                                                g.grad.shape[0], -1, g.grad.shape[-1]
                                            )
                                            for g in grid
                                        ],
                                        dim=1,
                                    )
                                }
                            )
                        elif isinstance(grid, torch.Tensor):
                            grad_tensors.update({"grid": grid})
                        else:
                            raise NotImplementedError("no such grid type")

                        for v in grad_tensors.values():
                            assert torch.isfinite(v).all()

                        grads["naive" if is_naive else "lightplane"] = grad_tensors

                    if self.verbose:
                        print(f" backward check:")
                    test_results = [
                        compare_one(
                            grads["lightplane"][k],
                            grads["naive"][k],
                            "  " + k,
                            run_assert=False,
                            verbose=self.verbose,
                        )
                        for k in grads["naive"].keys()
                    ]
                    if not all(test_results):
                        assert False


if __name__ == "__main__":
    unittest.main(failfast=DEBUG)
