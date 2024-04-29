# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
import os
import sys
import functools
from logging import getLogger
from typing import Optional

from cogapp import Cog

import lightplane

from .templates import cog_util

logger = getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, "templates")
GENERATED_DIR = os.path.join(CURRENT_DIR, "generated")
SHARED_DIR = os.path.join(CURRENT_DIR, "shared")
TEMPLATE_RENDERER_FILE_NAMES = ("renderer_mlp_util", "renderer_fw", "renderer_bw")
TEMPLATE_SPLATTER_FILE_NAMES = ("splatter_mlp_util", "splatter_fw", "splatter_bw")


def get_lightplane_kernels(
    kernel_name: str,
    n_layers_trunk: int,
    n_layers_opacity: Optional[int] = None,
    n_layers_color: Optional[int] = None,
    regenerate_code: bool = False,
    cached_kernel_functors: bool = True,
):
    assert kernel_name in [
        "splatter",
        "renderer",
    ]  # kernel should be either splatter or renderer
    if kernel_name == "renderer":
        assert n_layers_opacity is not None
        assert n_layers_color is not None

    if cached_kernel_functors:
        if regenerate_code:
            _get_lightplane_kernels_cached.cache_clear()
        return _get_lightplane_kernels_cached(
            kernel_name,
            n_layers_trunk,
            n_layers_opacity,
            n_layers_color,
        )

    return _get_lightplane_kernels(
        kernel_name,
        n_layers_trunk,
        n_layers_opacity,
        n_layers_color,
        regenerate_code,
    )


@functools.lru_cache(maxsize=None)
def _get_lightplane_kernels_cached(
    kernel_name: str,
    n_layers_trunk: int,
    n_layers_opacity: Optional[int],
    n_layers_color: Optional[int],
):
    return _get_lightplane_kernels(
        kernel_name,
        n_layers_trunk,
        n_layers_opacity,
        n_layers_color,
        False,
    )


def _get_lightplane_kernels(
    kernel_name: str,
    n_layers_trunk: int,
    n_layers_opacity: Optional[int],
    n_layers_color: Optional[int],
    regenerate_code: bool,
):
    # Check for the last file change in the whole triton source folder.
    last_file_change = _get_last_lightplane_file_changes()

    # 1) check whether a cached version of the kernels exists, if not generate the code
    fw_file, bw_file = _generate_lighplane_code(
        kernel_name,
        n_layers_trunk,
        n_layers_opacity,
        n_layers_color,
        regenerate_code,
        last_file_change,
    )
    # 2) import the kernel code from the generated files
    fw_module = _import_module_filepath("fw", fw_file)
    bw_module = _import_module_filepath("bw", bw_file)

    if kernel_name == "renderer":
        return fw_module.fw_kernel, bw_module.bw_kernel
    elif kernel_name == "splatter":
        if (n_layers_trunk is None) or (n_layers_trunk == 0):
            return (
                fw_module.fw_kernel,
                fw_module.fw_kernel,
                bw_module.bw_kernel,
                bw_module.bw_kernel,
            )
        else:
            return (
                fw_module.fw_kernel_wMLP,
                fw_module.fw_kernel,
                bw_module.bw_kernel_wMLP,
                bw_module.bw_kernel,
            )
    else:
        raise NotImplementedError(
            "No such kernel!! kernel should be either splatter or renderer"
        )


def _get_last_lightplane_file_changes():
    last_change = -1
    for folder_path in [TEMPLATE_DIR, SHARED_DIR]:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    last_change = max(last_change, os.path.getmtime(full_path))
    return last_change


def _import_module_filepath(module_name, filepath):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generate_lighplane_code(
    kernel_name: str,
    n_layers_trunk: int,
    n_layers_opacity: Optional[int],
    n_layers_color: Optional[int],
    regenerate_code: bool,
    last_file_change: float,
):
    fw_file, bw_file = None, None

    if kernel_name == "splatter":
        _template_file = TEMPLATE_SPLATTER_FILE_NAMES
    elif kernel_name == "renderer":
        _template_file = TEMPLATE_RENDERER_FILE_NAMES
    else:
        raise NotImplementedError("No such kernel name!")

    for template_name in _template_file:
        generated_file = _generate_lightplane_file(
            kernel_name,
            template_name,
            n_layers_trunk,
            n_layers_opacity,
            n_layers_color,
            regenerate_code,
            last_file_change,
        )
        if template_name == "renderer_fw" or template_name == "splatter_fw":
            fw_file = generated_file
        elif template_name == "renderer_bw" or template_name == "splatter_bw":
            bw_file = generated_file

    return fw_file, bw_file


def _generate_lightplane_file(
    kernel_name: str,
    template_name: str,
    n_layers_trunk: int,
    n_layers_opacity: Optional[int],
    n_layers_color: Optional[int],
    regenerate_code: bool,
    last_file_change: float,
):
    if kernel_name == "renderer":
        generated_file = os.path.join(
            GENERATED_DIR,
            cog_util.get_generated_file_name(
                template_name,
                n_layers_trunk,
                n_layers_opacity,
                n_layers_color,
            )
            + ".py",
        )
    elif kernel_name == "splatter":
        generated_file = os.path.join(
            GENERATED_DIR,
            cog_util.get_generated_splatter_file_name(template_name, n_layers_trunk)
            + ".py",
        )
    else:
        raise NotImplementedError("No such kernel name!")

    template_file = os.path.join(TEMPLATE_DIR, template_name + ".py")

    templates_newer_than_generated = os.path.exists(generated_file) and (
        last_file_change > os.path.getmtime(generated_file)
    )

    if templates_newer_than_generated:
        logger.warning(
            "Changes in the template code detected -> regenerating triton code."
        )

    if (
        regenerate_code
        or (not os.path.exists(generated_file))
        or templates_newer_than_generated
    ):
        if kernel_name == "renderer":
            mlp_util_file = os.path.join(
                GENERATED_DIR,
                cog_util.get_generated_file_name(
                    "renderer_mlp_util",
                    n_layers_trunk,
                    n_layers_opacity,
                    n_layers_color,
                )
                + ".py",
            )
        elif kernel_name == "splatter":
            mlp_util_file = os.path.join(
                GENERATED_DIR,
                cog_util.get_generated_splatter_file_name(
                    "splatter_mlp_util", n_layers_trunk
                )
                + ".py",
            )
        else:
            raise NotImplementedError("No such kernel name!")

        lightplane_path = os.path.dirname(lightplane.__path__[0])
        rel_mlp_util_module, rel_cog_util_module = (
            p.replace(os.path.sep, ".").replace(".py", "")
            for p in [
                os.path.relpath(mlp_util_file, lightplane_path),
                os.path.relpath(cog_util.__file__, lightplane_path),
            ]
        )

        if kernel_name == "renderer":
            logger.warning(
                f"Generating triton code for `{template_name}.py`, generation parameters:\n"
                + f"    N_LAYERS_TRUNK={n_layers_trunk}\n"
                + f"    N_LAYERS_OPACITY={n_layers_opacity}\n"
                + f"    N_LAYERS_COLOR={n_layers_color}\n"
            )

            code_generator = Cog()
            code_generator.main(
                [
                    None,
                    "-o",
                    generated_file,
                    "-d",
                    "-D",
                    f"N_LAYERS_TRUNK={n_layers_trunk}",
                    "-D",
                    f"N_LAYERS_OPACITY={n_layers_opacity}",
                    "-D",
                    f"N_LAYERS_COLOR={n_layers_color}",
                    "-D",
                    f"COG_UTIL_MODULE={rel_cog_util_module}",
                    "-D",
                    f"MLP_UTIL_MODULE={rel_mlp_util_module}",
                    template_file,
                ]
            )
        elif kernel_name == "splatter":
            logger.warning(
                f"Generating triton code for `{template_name}.py`, generation parameters:\n"
                + f"    N_LAYERS={n_layers_trunk}\n"
            )

            code_generator = Cog()
            code_generator.main(
                [
                    None,
                    "-o",
                    generated_file,
                    "-d",
                    "-D",
                    f"N_LAYERS={n_layers_trunk}",
                    "-D",
                    f"COG_UTIL_MODULE={rel_cog_util_module}",
                    "-D",
                    f"MLP_UTIL_MODULE={rel_mlp_util_module}",
                    template_file,
                ]
            )

        logger.warning(f"Generated {generated_file}")

    return generated_file
