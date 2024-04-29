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
    get_test_example,
    get_test_grid_size,
    random_mlp_decoder_params,
    seed_random_generators,
)

from lightplane import Rays, lightplane_renderer, lightplane_renderer_naive
from lightplane.mlp_utils import DecoderParams, flattened_decoder_params_to_list

USE_TENSOR_GRID = False  # whether the input grid is list of tensor or flatten tensor


def print_memory_usage():
    stats = torch.cuda.memory_stats()
    stats = {
        s: v
        for s, v in stats.items()
        if any(
            p in s
            for p in [
                ".all."
                # ".all.allocated"
                # ".all.peak"
                # ".all.freed"
            ]
        )
    }
    print("---")
    for k, v in stats.items():
        print(f"{k}={v}")


def _test_one_full(p_dict, n_reruns, n_warmup):
    locals_ = locals()
    test_signature = "Test:\n  " + "\n  ".join(
        f"{k}={str(v)}" for k, v in locals_.items() if k not in ["near", "far"]
    )
    print(test_signature)
    stats = {k: v for k, v in locals_.items()}
    num_rays = p_dict["n_rays"]
    stats["image_size"] = int(num_rays**0.5)
    stats["num_rays"] = num_rays

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
        ]
    }

    for mode in ["checkpoint", "naive", "kernel"]:
        lighplane_kwargs = dict(
            num_samples=p_dict["num_samples"],
            num_samples_inf=p_dict["num_samples_inf"],
            gain=p_dict["gain"],
            mask_out_of_bounds_samples=p_dict["mask_out_of_bounds_samples"],
            inject_noise_sigma=p_dict["inject_noise_sigma"],
            contract_coords=p_dict["contract_coords"],
            inject_noise_seed=0,
            disparity_at_inf=0.01,
            triton_block_size=16,
            triton_num_warps=4,
        )

        if mode == "naive":
            test_name = "naive"
            fun = lightplane_renderer_naive
        elif mode == "kernel":
            test_name = "kernel"
            fun = lightplane_renderer
        elif mode == "checkpoint":
            test_name = "checkpoint"
            fun = lightplane_renderer_naive
            lighplane_kwargs.update({"checkpointing": True})

        perf_stats = []
        for it in range(n_reruns + n_warmup):
            perf_dict = {}

            def do_init_buffers(requires_grad):
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
                        requires_grad=requires_grad,
                        use_tensor_grid=USE_TENSOR_GRID,
                    )

                    return {
                        "rays": rays,
                        "grid": grid,
                        "decoder_params": mlp_params,
                        "color_grid": color_grid,
                        "scaffold": scaffold,
                        "grid_sizes": grid_sizes,
                        "color_grid_sizes": color_grid_sizes,
                    }

            # with torch.no_grad():
            # forward pass should have gradient enabled, so that intermediate results would be stored.
            fw_inputs = do_init_buffers(True)
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
                    # # enable grads
                    # for k, v in fw_inputs.items():
                    #     if torch.is_tensor(v):
                    #         v.requires_grad = True
                    # with torch.no_grad():

                    # run one more time to fill up the grad buffer
                    out = fun(
                        **fw_inputs,
                        **lighplane_kwargs,
                    )
                    grad_var = sum((o * torch.randn_like(o)).mean() for o in out)
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


def renderer_speed_benchmark():
    torch.manual_seed(0)

    stats_print_keys = [
        "num_rays",
        "image_size",
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
        "mask_out_of_bounds_samples": [False],
        "contract_coords": [False],
        "grid_size": [[3, 32, 32, 32, 32]],
        "one_ray": [False],
        "n_hidden": [32],
        "num_samples": [256],
        "num_samples_inf": [0],
        "gain": [1.0],
        "n_rays": [0],
        "color_chn": [3],
        "scaffold_size": [None],
        "grid_size_color": [None],
        "is_triplane": [True],
        "n_layers_trunk": [2],
        "n_layers_opacity": [2],
        "n_layers_color": [2],
        "inject_noise_sigma": [0.0],
    }
    n_reruns = 5
    n_warmup = 2
    test_sweep_nums = {k: list(range(len(v))) for k, v in test_sweep.items()}

    for p, np in zip(
        tqdm(list(itertools.product(*list(test_sweep.values())))),
        list(itertools.product(*list(test_sweep_nums.values()))),
    ):
        print("STATS," + ",".join(stats_print_keys))

        p_dict = copy.deepcopy(dict(zip(test_sweep.keys(), p)))
        num_p_dict = copy.deepcopy(dict(zip(test_sweep.keys(), np)))
        use_separate_color_grid = p_dict["grid_size_color"] is not None
        if use_separate_color_grid:
            p_dict["n_layers_trunk"] = 0
        print("test params:")
        for k, v in p_dict.items():
            print(
                f"  {k:30s}: {str(v):30s} ({num_p_dict[k] + 1} / {len(test_sweep[k])})"
            )
        all_stats = []

        for im_size in (
            16,
            16 * math.sqrt(2),
            32,
            32 * math.sqrt(2),
            64,
            64 * math.sqrt(2),
            128,
            128 * math.sqrt(2),
            256,
            256 * math.sqrt(2),
            512,
            512 * math.sqrt(2),
            1024,
            1024 * math.sqrt(2),
            2048,
        ):
            num_rays = int(im_size**2)
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
    renderer_speed_benchmark()
