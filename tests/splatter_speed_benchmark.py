# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# %%
import copy
import itertools
import math
import sys
import time

import torch
import triton
import triton.language as tl
from tqdm import tqdm
from utils import (
    Memory,
    Timer,
    compare_one,
    get_test_example_splatter,
    lightplane_splatter_test_warpper,
    random_mlp_decoder_params,
    seed_random_generators,
)

from lightplane import (
    Rays,
    lightplane_mlp_splatter,
    lightplane_splatter,
    lightplane_splatter_naive,
)

USE_TENSOR_GRID = False  # whether the input grid is list of tensor or flatten tensor


def _test_one_full(p_dict, n_reruns, n_warmup):
    locals_ = locals()
    test_signature = "Test:\n  " + "\n  ".join(
        f"{k}={str(v)}" for k, v in locals_.items() if k not in ["near", "far"]
    )
    print(test_signature)
    stats = {k: v for k, v in locals_.items()}

    stats["num_views"] = p_dict["num_view"]
    stats["num_rays"] = p_dict["n_rays"]

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
        ]
    }

    for mode in ["checkpoint", "naive", "kernel"]:
        lighplane_kwargs = dict(
            num_samples=p_dict["num_samples"],
            num_samples_inf=p_dict["num_samples_inf"],
            mask_out_of_bounds_samples=p_dict["mask_out_of_bounds_samples"],
            contract_coords=p_dict["contract_coords"],
            disparity_at_inf=0.01,
            triton_block_size=16,
            triton_num_warps=4,
            return_list=False,
        )

        fun = lightplane_splatter_test_warpper
        if mode == "naive":
            test_name = "naive"
            lighplane_kwargs.update({"use_naive_implementation": True})
        elif mode == "kernel":
            test_name = "kernel"
        elif mode == "checkpoint":
            test_name = "checkpoint"
            lighplane_kwargs.update(
                {"use_naive_implementation": True, "checkpointing": True}
            )

        perf_stats = []
        for it in range(n_reruns + n_warmup):
            perf_dict = {}

            def do_init_buffers(requires_grad):
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
                        requires_grad=requires_grad,
                        use_tensor_grid=USE_TENSOR_GRID,
                    )
                    rays.encoding = splatting_feature
                    return {
                        "rays": rays,
                        "output_grid_size": grid_size,
                        "mlp_params": mlp_params,
                        "input_grid": input_grid,
                        "input_grid_sizes": input_grid_sizes,
                    }

            fw_inputs = do_init_buffers(True)
            with torch.no_grad():
                with Memory("fw_" + test_name) as mem:
                    try:
                        with Timer("fw_" + test_name) as tm:
                            fun(
                                **fw_inputs,
                                **lighplane_kwargs,
                            )
                    except torch.cuda.OutOfMemoryError:
                        print("OOM!")
                        continue

            perf_dict.update(tm.get_stats_dict())
            perf_dict.update(mem.get_stats_dict())

            del fw_inputs

            fw_inputs = do_init_buffers(True)
            with Memory("bw_" + test_name) as mem:
                try:
                    out = fun(
                        **fw_inputs,
                        **lighplane_kwargs,
                    )
                    grad_var = torch.sum(out)

                    # run one more time to fill up the grad buffer
                    with Timer("bw_" + test_name) as tm:
                        grad_var.backward()
                except torch.cuda.OutOfMemoryError:
                    print("OOM!")
                    continue

            perf_dict.update(tm.get_stats_dict())
            perf_dict.update(mem.get_stats_dict())
            if it > n_warmup:
                perf_stats.append(perf_dict)

        if len(perf_stats) > 0:
            for k in perf_stats[0].keys():
                stats[k] = sum(s[k] for s in perf_stats) / len(perf_stats)

    return stats


def splatter_speed_benchmark():
    torch.manual_seed(0)

    stats_print_keys = [
        "num_views",
        "num_rays",
        # ----
        "t_fw_kernel",
        "t_fw_naive",
        "t_fw_checkpoint",
        "t_bw_kernel",
        "t_bw_naive",
        "t_bw_checkpoint",
        # ----
        # "num_rays",
        # "image_size",
        "mem_fw_kernel",
        "mem_bw_kernel",
        "mem_fw_naive",
        "mem_bw_naive",
        "mem_fw_checkpoint",
        "mem_bw_checkpoint",
        # ----
        # "num_rays",
        # "image_size",
        "max_mem_fw_kernel",
        "max_mem_bw_kernel",
        "max_mem_fw_naive",
        "max_mem_bw_naive",
        "max_mem_fw_checkpoint",
        "max_mem_bw_checkpoint",
    ]

    report_lines = []
    report_lines.append("STATS," + ",".join(stats_print_keys))
    print(report_lines[-1])

    test_sweep = {
        "mask_out_of_bounds_samples": [True],
        "contract_coords": [False],
        "grid_size": [[3, 160, 160, 160, 64]],
        "input_grid_size": [[3, 160, 160, 160, 64]],
        "one_ray": [False],
        "n_rays": [128 * 128],
        "num_samples": [96],
        "num_samples_inf": [0],
        "is_triplane": [False],
        "use_mlp": [False],
        "n_hidden": [64],
        "n_layers": [2],
        "feat_dim": [64],
    }
    n_reruns = 5
    n_warmup = 2
    im_size = 128
    test_sweep_nums = {k: list(range(len(v))) for k, v in test_sweep.items()}

    for p, np in zip(
        tqdm(list(itertools.product(*list(test_sweep.values())))),
        list(itertools.product(*list(test_sweep_nums.values()))),
    ):
        print("STATS," + ",".join(stats_print_keys))

        p_dict = copy.deepcopy(dict(zip(test_sweep.keys(), p)))
        num_p_dict = copy.deepcopy(dict(zip(test_sweep.keys(), np)))

        # for fair, we only render from a single voxel grid or a set of triplanes
        if not p_dict["is_triplane"]:
            p_dict["grid_size"][0] = 1
            p_dict["input_grid_size"][0] = 1
        else:
            p_dict["grid_size"][0] = 3
            p_dict["input_grid_size"][0] = 3

        if p_dict["use_mlp"] is False:
            mlp_n_layers = 0
        else:
            mlp_n_layers = p_dict["n_layers"]

        print("test params:")
        for k, v in p_dict.items():
            print(
                f"  {k:30s}: {str(v):30s} ({num_p_dict[k] + 1} / {len(test_sweep[k])})"
            )
        all_stats = []
        for num_view in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            num_rays = int(im_size**2) * num_view
            p_dict["num_view"] = num_view
            p_dict["n_rays"] = num_rays
            stats = _test_one_full(
                p_dict,
                n_reruns,
                n_warmup,
            )
            all_stats.append(stats)
            report_lines.append(
                "STATS,"
                + ",".join(
                    [f"{stats.get(k, float('NaN')):1.5f}" for k in stats_print_keys]
                )
            )
            print(report_lines[-1])

    for l in report_lines:
        print(l)


if __name__ == "__main__":
    splatter_speed_benchmark()
