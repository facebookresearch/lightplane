# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json

import configargparse
import torch

from examples.utils.dataloader.dataset import datasets


def define_common_args(parser: configargparse.ArgumentParser):
    parser.add_argument("--data_dir", type=str)

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Config file (will override args)",
    )

    group = parser.add_argument_group("Data loading")
    group.add_argument(
        "--dataset_type",
        choices=list(datasets.keys()) + ["auto"],
        default="auto",
        help="Dataset type (specify type or use auto)",
    )
    group.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="Number of training images. Defaults to use all avaiable.",
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
    group.add_argument("--factor", type=int, default=4)

    group = parser.add_argument_group("3D Repre. options")
    group.add_argument(
        "--grid_representation",
        type=str,
        default="voxel_grid",
        help="3D representation used for optimization: voxel_grid or triplane",
    )
    group.add_argument(
        "--grid_resolution",
        type=int,
        default=64,
        help="resolution used in 3D representation",
    )
    group.add_argument(
        "--grid_feat_dim",
        type=int,
        default=32,
        help="the feature channel of 3D representations",
    )
    group.add_argument(
        "--upsample_step",
        type=list,
        default=[100000, 100000, 100000],
        help="exceeding this step, will double the voxel size",
    )

    group = parser.add_argument_group("Rendering options")
    group.add_argument(
        "--renderer_type",
        type=str,
        default="image",
        choices=["image", "rays"],
        help="render full image or rays",
    )
    group.add_argument(
        "--n_rays", type=int, default=4096, help="ray numbers for render rays"
    )
    group.add_argument(
        "--near", type=float, default=0.1, help="near for point sampling"
    )
    group.add_argument("--far", type=float, default=2.0, help="far for point sampling")
    group.add_argument(
        "--num_samples",
        type=int,
        default=256,
        help="sample points number along the ray",
    )
    group.add_argument(
        "--num_samples_inf",
        type=int,
        default=64,
        help="sampling points for infinite far",
    )
    group.add_argument(
        "--contract_coords",
        action="store_true",
        default=False,
        help="whether use contract coordinates",
    )
    group.add_argument(
        "--disparity_at_inf", type=float, default=1e-4, help="disparity at infinite far"
    )
    group.add_argument(
        "--mlp_hidden_chn", type=int, default=32, help="mlp dimensions for renderer"
    )
    group.add_argument("--mlp_n_layers_opacity", type=int, default=1)
    group.add_argument("--mlp_n_layers_trunk", type=int, default=1)
    group.add_argument("--mlp_n_layers_color", type=int, default=2)
    group.add_argument("--use_naive_impl", action="store_true", default=False)
    group.add_argument("--ray_embedding_num_harmonics", type=int, default=2)
    group.add_argument("--use_scaffold", action="store_true", default=False)
    group.add_argument("--scaffold_size", type=int, default=128)
    group.add_argument("--update_scaffold_step", type=list, default=[2000])

    group = parser.add_argument_group("display options")
    group.add_argument(
        "--progress_refresh_rate", type=int, default=10, help="refresh rate for tqdm"
    )
    group.add_argument(
        "--eval_rate", type=int, default=2000, help="iteration number for evaluation"
    )

    group = parser.add_argument_group("optimizer")
    group.add_argument("--num_iters", type=int, default=30000, help="optimizing steps")
    group.add_argument(
        "--lr_grids",
        type=float,
        default=0.01,
        help="learning rate for hashed representation (grids)",
    )
    group.add_argument(
        "--lr_nn", type=float, default=0.001, help="learning rate for neural network"
    )
    group.add_argument(
        "--lr_decay_iters", type=int, default=-1, help="learning rate decay iterations"
    )
    group.add_argument(
        "--lr_decay_target_ratio",
        type=float,
        default=0.1,
        help="learning rate decay target",
    )
    group.add_argument(
        "--lr_upsample_reset",
        action="store_true",
        default=False,
        help="whether reset learning rate after upsmapling",
    )
    group.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    group.add_argument("--beta2", type=float, default=0.99, help="beta1 for adam")
    group.add_argument(
        "--tv_loss_weight", type=float, default=-1.0, help="weight of tv loss"
    )
    group.add_argument(
        "--l1_loss_weight", type=float, default=-1.0, help="weight for L1 loss"
    )
    group.add_argument(
        "--lpips_loss_weight", type=float, default=0.001, help="weight for lpips loss"
    )


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
    Load a json config file if specified and merge the arguments
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
