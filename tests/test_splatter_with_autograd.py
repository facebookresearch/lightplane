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
from utils import (
    compare_one,
    get_test_example_splatter,
    lightplane_splatter_test_warpper,
    seed_random_generators,
)

# If True, produces a verbose stdout-output, and stops unittest after the first failure.
DEBUG = False


class TestSplatterWithAutograd(unittest.TestCase):
    def setUp(self, verbose=DEBUG):
        seed_random_generators(0)
        self.verbose = verbose

    def test(self):
        test_sweep = {
            "contract_coords": [False, True],
            "mask_out_of_bounds_samples": [False, True],
            "grid_size": [[2, 16, 12, 8, 32]],
            "input_grid_size": [None, [2, 10, 14, 16, 32]],
            "one_ray": [False],
            "num_samples": [16],
            "num_samples_inf": [11],
            "n_rays": [1, 128],
            "is_triplane": [False, True],
            "use_mlp": [True, False],
            "n_hidden": [64],
            "n_layers": [3, 4],
            "feat_dim": [64, 32],
            "use_tensor_grid": [True, True],
        }

        test_sweep_nums = {k: list(range(len(v))) for k, v in test_sweep.items()}

        for p, np in zip(
            tqdm(list(itertools.product(*list(test_sweep.values())))),
            list(itertools.product(*list(test_sweep_nums.values()))),
        ):
            for it in range(5):
                p_dict = copy.deepcopy(dict(zip(test_sweep.keys(), p)))
                num_p_dict = copy.deepcopy(dict(zip(test_sweep.keys(), np)))

                if not p_dict["use_mlp"] and p_dict["input_grid_size"] is not None:
                    continue  # this combo does not make sense

                if p_dict["input_grid_size"] is not None:
                    if p_dict["feat_dim"] != p_dict["input_grid_size"][-1]:
                        p_dict["input_grid_size"][-1] = p_dict[
                            "feat_dim"
                        ]  # splatting feature should have the same dimension as input grid; we can always change feature dimensions by MLP.

                if (p_dict["grid_size"][-1] != p_dict["feat_dim"]) and (
                    p_dict["use_mlp"] is False
                ):
                    continue  # this combo does not make sense: when not using any MLP, the splatting feature dimension should be the same as target grid feature dimension.

                if p_dict["use_tensor_grid"] and p_dict["input_grid_size"] is None:
                    continue  # doesn't make sense

                test_example_params = {
                    k: p_dict[k]
                    for k in [
                        "grid_size",
                        "n_rays",
                        "feat_dim",
                        "use_mlp",
                        "n_hidden",
                        "n_layers",
                        "input_grid_size",
                        "is_triplane",
                        "one_ray",
                        "use_tensor_grid",
                    ]
                }

                with torch.random.fork_rng():
                    seed_random_generators(it)
                    (
                        rays,
                        grid_size,
                        splatting_feature,
                        mlp_params,
                        input_grid,
                        input_grid_sizes,
                    ) = get_test_example_splatter(
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
                    rays.encoding = splatting_feature
                    lighplane_splatter_args = [
                        rays,
                        grid_size,
                    ]
                    lighplane_splatter_kwargs = dict(
                        num_samples=p_dict["num_samples"],
                        num_samples_inf=p_dict["num_samples_inf"],
                        mask_out_of_bounds_samples=p_dict["mask_out_of_bounds_samples"],
                        contract_coords=p_dict["contract_coords"],
                        disparity_at_inf=0.01,
                        triton_block_size=16,
                        triton_num_warps=4,
                        input_grid_sizes=input_grid_sizes,
                        mlp_params=mlp_params,
                        input_grid=input_grid,
                    )
                    splatting_results_ = lightplane_splatter_test_warpper(
                        *lighplane_splatter_args,
                        **lighplane_splatter_kwargs,
                        use_naive_implementation=True,
                    )
                    splatting_results = lightplane_splatter_test_warpper(
                        *lighplane_splatter_args,
                        **lighplane_splatter_kwargs,
                        use_naive_implementation=False,
                    )

                    # cat the output splatting grids to a single tensor
                    splatting_results, splatting_results_ = (
                        torch.cat([g.view(-1, g.shape[-1]) for g in sr], dim=0)
                        for sr in [splatting_results, splatting_results_]
                    )

                    if self.verbose:
                        print(f" forward check:")
                    test_results = [
                        compare_one(
                            splatting_results,
                            splatting_results_,
                            " splatting_results",
                            run_assert=False,
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
                                grid_size,
                                splatting_feature,
                                mlp_params,
                                input_grid,
                                input_grid_sizes,
                            ) = get_test_example_splatter(
                                **test_example_params,
                                requires_grad=True,
                            )
                        rays.encoding = splatting_feature
                        lighplane_splatter_args = [
                            rays,
                            grid_size,
                        ]
                        lighplane_splatter_kwargs = dict(
                            num_samples=p_dict["num_samples"],
                            num_samples_inf=p_dict["num_samples_inf"],
                            mask_out_of_bounds_samples=p_dict[
                                "mask_out_of_bounds_samples"
                            ],
                            contract_coords=p_dict["contract_coords"],
                            disparity_at_inf=0.01,
                            input_grid_sizes=input_grid_sizes,
                            triton_block_size=16,
                            triton_num_warps=4,
                            mlp_params=mlp_params,
                            input_grid=input_grid,
                        )
                        results = lightplane_splatter_test_warpper(
                            *lighplane_splatter_args,
                            **lighplane_splatter_kwargs,
                            use_naive_implementation=is_naive,
                        )
                        results = torch.cat(
                            [g.view(-1, g.shape[-1]) for g in results], dim=0
                        )
                        with torch.random.fork_rng():
                            seed_random_generators(it)
                            loss = torch.sum(torch.randn_like(results) * results)

                        loss.backward()

                        grad_tensors = {
                            "splatting_feature": splatting_feature.grad,
                        }
                        if test_example_params["use_mlp"]:
                            if isinstance(input_grid, list):
                                grad_tensors.update(
                                    {
                                        "grid": torch.cat(
                                            [
                                                g.grad.reshape(
                                                    g.grad.shape[0],
                                                    -1,
                                                    g.grad.shape[-1],
                                                )
                                                for g in input_grid
                                            ],
                                            dim=1,
                                        )
                                    }
                                )
                            elif isinstance(input_grid, torch.Tensor):
                                grad_tensors.update({"grid": input_grid})
                            else:
                                raise NotImplementedError("no such grid type")

                            grad_tensors.update(
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
                                }
                            )
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
