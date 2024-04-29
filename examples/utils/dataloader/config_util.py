# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json

import torch
from util.dataset import datasets


def define_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("data_dir", type=str)

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Config yaml file (will override args)",
    )

    group = parser.add_argument_group("Data loading")
    group.add_argument(
        "--dataset_type",
        choices=list(datasets.keys()) + ["auto"],
        default="auto",
        help="Dataset type (specify type or use auto)",
    )
    group.add_argument(
        "--scene_scale",
        type=float,
        default=None,
        help="Global scene scaling (or use dataset default)",
    )
    group.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Image scale, e.g. 0.5 for half resolution (or use dataset default)",
    )
    group.add_argument(
        "--seq_id", type=int, default=1000, help="Sequence ID (for CO3D only)"
    )
    group.add_argument(
        "--epoch_size",
        type=int,
        default=12800,
        help="Pseudo-epoch size in term of batches (to be consistent across datasets)",
    )
    group.add_argument(
        "--white_bkgd",
        type=bool,
        default=True,
        help="Whether to use white background (ignored in some datasets)",
    )
    group.add_argument("--llffhold", type=int, default=8, help="LLFF holdout every")
    group.add_argument(
        "--normalize_by_bbox",
        type=bool,
        default=False,
        help="Normalize by bounding box in bbox.txt, if available (NSVF dataset only); precedes normalize_by_camera",
    )
    group.add_argument(
        "--data_bbox_scale",
        type=float,
        default=1.2,
        help="Data bbox scaling (NSVF dataset only)",
    )
    group.add_argument(
        "--cam_scale_factor",
        type=float,
        default=0.95,
        help="Camera autoscale factor (NSVF/CO3D dataset only)",
    )
    group.add_argument(
        "--normalize_by_camera",
        type=bool,
        default=True,
        help="Normalize using cameras, assuming a 360 capture (NSVF dataset only); only used if not normalize_by_bbox",
    )
    group.add_argument(
        "--perm",
        action="store_true",
        default=False,
        help="sample by permutation of rays (true epoch) instead of "
        "uniformly random rays",
    )

    group = parser.add_argument_group("Render options")


def build_data_options(args):
    """
    Arguments to pass as kwargs to the dataset constructor
    """
    return {
        "dataset_type": args.dataset_type,
        "seq_id": args.seq_id,
        "epoch_size": args.epoch_size * args.__dict__.get("batch_size", 5000),
        "scene_scale": args.scene_scale,
        "scale": args.scale,
        "white_bkgd": args.white_bkgd,
        "hold_every": args.llffhold,
        "normalize_by_bbox": args.normalize_by_bbox,
        "data_bbox_scale": args.data_bbox_scale,
        "cam_scale_factor": args.cam_scale_factor,
        "normalize_by_camera": args.normalize_by_camera,
        "permutation": args.perm,
    }


def maybe_merge_config_file(args, allow_invalid=False):
    """
    Load json config file if specified and merge the arguments
    """
    if args.config is not None:
        with open(args.config) as config_file:
            configs = json.load(config_file)
        invalid_args = list(set(configs.keys()) - set(dir(args)))
        if invalid_args and not allow_invalid:
            raise ValueError(f"Invalid args {invalid_args} in {args.config}.")
        args.__dict__.update(configs)


def setup_render_opts(opt, args):
    """
    Pass render arguments to the SparseGrid renderer options
    """
    opt.step_size = args.step_size
    opt.sigma_thresh = args.sigma_thresh
    opt.stop_thresh = args.stop_thresh
    opt.background_brightness = args.background_brightness
    opt.backend = args.renderer_backend
    opt.random_sigma_std = args.random_sigma_std
    opt.random_sigma_std_background = args.random_sigma_std_background
    opt.last_sample_opaque = args.last_sample_opaque
    opt.near_clip = args.near_clip
    opt.use_spheric_clip = args.use_spheric_clip
