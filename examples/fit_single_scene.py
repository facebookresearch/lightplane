# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import math
import os
import random
import sys
import time

import configargparse
import imageio
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from utils.dataloader.dataset import datasets
from utils.util import config_util, io_util
from utils.util.grid_util import grid_L1_loss, grid_TV_loss, grid_up_sample
from utils.util.metric import PSNR_metric, rgb_lpips, rgb_ssim

from lightplane import LightplaneRenderer, Rays


def get_argparse(cmd=None):
    parser = configargparse.ArgumentParser(
        description="Fit a single NeRF model to a scene"
    )
    config_util.define_common_args(parser)

    group = parser.add_argument_group("general")
    group.add_argument("--random_seed", type=int, default=20240209)
    group.add_argument(
        "--save_dir",
        "-t",
        type=str,
        default="ckpt",
        help="checkpoint and logging directory",
    )
    group.add_argument("--init_ckpt", type=str, default=None, help="")
    if cmd is not None:
        args = parser.parse_args(cmd)
    else:
        args = parser.parse_args()
    config_util.maybe_merge_config_file(args)

    return args


@torch.no_grad()
def evaluation(
    dset,
    device,
    eval_num,
    args,
    render_module,
    grid,
    scaffold,
    logging_path,
    model_path,
    prefix,
    save_txt=False,
    eval_extra_metrics=False,
):
    """
    Evaluation function for rendering images from pretrained field.
    """
    N_IMGS_TO_EVAL = min(dset.n_images, eval_num)
    N_IMGS_TO_SAVE = N_IMGS_TO_EVAL  # if not args.tune_mode else 1
    img_eval_interval = dset.n_images // N_IMGS_TO_EVAL
    img_save_interval = N_IMGS_TO_EVAL // N_IMGS_TO_SAVE
    img_ids = range(0, dset.n_images, img_eval_interval)

    n_images_gen = 0
    PSNRs_test = []
    ssims, l_alex, l_vgg = [], [], []
    for i, img_id in enumerate(img_ids):
        image_size = dset.get_image_size(img_id)
        near_t = torch.ones(image_size, device=device).view(-1) * args.near
        far_t = torch.ones(image_size, device=device).view(-1) * args.far
        ray_size = far_t.shape[0]
        rays = Rays(
            directions=dset.rays.dirs[img_id].to(device),
            origins=dset.rays.origins[img_id].to(device),
            grid_idx=torch.zeros(ray_size, device=device, dtype=torch.long),
            near=near_t,
            far=far_t,
        )
        (
            ray_length_render,
            alpha_render,
            feature_render,
        ) = render_module(rays=rays, feature_grid=grid, scaffold=scaffold)

        mse_loss = torch.nn.functional.mse_loss(
            feature_render, dset.rays.gt[img_id].to(device)
        )

        img_fl, gt_img_fl, depth_fl = (
            os.path.join(logging_path, fl)
            for fl in (
                f"{prefix}_{i}_render.png",
                f"{prefix}_{i}_gt.png",
                f"{prefix}_{i}_ray_length.png",
            )
        )
        print(f"Outputting visualisation to {img_fl}, {gt_img_fl}, {depth_fl}")

        # save rgb render, rgb ground truth, ray-length render
        img = feature_render.view(image_size + (3,)).detach().cpu().numpy()
        imageio.imwrite(img_fl, np.uint8(img * 255.0))
        gt_img = dset.rays.gt[img_id].view(image_size + (3,)).numpy()
        imageio.imwrite(gt_img_fl, np.uint8(gt_img * 255.0))
        ray_length_img = io_util.convert_depth_image_to_colormap(
            ray_length_render.view(image_size)
        )
        imageio.imwrite(depth_fl, ray_length_img)

        PSNRs_test.append(-10.0 * np.log(mse_loss.item()) / np.log(10.0))
        if eval_extra_metrics:
            ssim = rgb_ssim(img, gt_img, 1)
            l_a = rgb_lpips(gt_img, img, "alex", device)
            l_v = rgb_lpips(gt_img, img, "vgg", device)
            ssims.append(ssim)
            l_alex.append(l_a)
            l_vgg.append(l_v)

    if save_txt:
        with open(os.path.join(os.path.dirname(model_path), "results.txt"), "w") as f:
            f.write(f"PSNR: {np.mean(PSNRs_test)} \n")
            if eval_extra_metrics:
                f.write(f"SSIM: {np.mean(ssims)} \n")
                f.write(f"LPIPS_Alex: {np.mean(l_alex)} \n")
                f.write(f"LPIPS_Vgg: {np.mean(l_vgg)} \n")

    save_module = {
        "grid": grid,
        "renderer": render_module.state_dict(),
        "scaffold": scaffold,
    }
    torch.save(save_module, model_path)
    return PSNRs_test


def main(args):
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logging path
    logging_path, model_path = io_util.get_save_path(args.save_dir)
    with open(os.path.join(args.save_dir, "exp_paramter.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # set up dataset for training (dset) and dataset for evaluation (dset_test)
    dset = datasets[args.dataset_type](
        args.data_dir,
        split="train",
        device=device,
        factor=args.factor,
        n_images=args.n_train,
        sample_image=(args.renderer_type == "image"),
        **config_util.build_data_options(args),
    )
    dset_test = datasets[args.dataset_type](
        args.data_dir,
        split="test",
        factor=args.factor,
        **config_util.build_data_options(args),
    )

    # initialize lightplane module
    num_channels = args.grid_feat_dim
    render_module = LightplaneRenderer(
        num_samples=args.num_samples,
        color_chn=3,
        grid_chn=num_channels,
        mlp_hidden_chn=args.mlp_hidden_chn,
        mlp_n_layers_opacity=args.mlp_n_layers_opacity,
        mlp_n_layers_trunk=args.mlp_n_layers_trunk,
        mlp_n_layers_color=args.mlp_n_layers_color,
        opacity_init_bias=-10.0,
        ray_embedding_num_harmonics=args.ray_embedding_num_harmonics,
        bg_color=1.0,
        contract_coords=args.contract_coords,
        num_samples_inf=args.num_samples_inf,
        disparity_at_inf=args.disparity_at_inf,
        use_naive_impl=args.use_naive_impl,
    ).to(device)

    # initialize grid representation
    vol_size = args.grid_resolution
    if args.grid_representation == "voxel_grid":
        # use voxel grid as representation
        v = torch.rand(1, vol_size, vol_size, vol_size, num_channels, device=device)
        v.requires_grad = True
        grid = [v]
    elif args.grid_representation == "triplane":
        # use triplane as representation
        grid = []
        for planei in range(3):
            size = [1, vol_size, vol_size, vol_size, num_channels]
            size[planei + 1] = 1
            triplane = (4 / pow(num_channels, 1 / 3)) * torch.rand(size, device=device)
            triplane.requires_grad = True
            grid.append(triplane)
    else:
        raise NotImplementedError("no such 3D Representation")

    # initialize scaffold
    use_scaffold = args.use_scaffold
    scaffold = None

    # load checkpoint
    if args.init_ckpt is not None:
        ckpt = torch.load(args.init_ckpt)
        with torch.no_grad():
            for index in range(len(ckpt["grid"])):
                grid[index].data = ckpt["grid"][index].data
            render_module.load_state_dict(ckpt["renderer"])
            scaffold = ckpt["scaffold"]

    # Set optimizer.  We give different learning rate for `grids and MLPs in `render_module`.
    optimizer = torch.optim.Adam(
        [
            {"params": grid, "lr": args.lr_grids},
            {
                "params": render_module.parameters(),
                "lr": args.lr_nn,
            },
        ],
        lr=args.lr_nn,
        betas=(args.beta1, args.beta2),
    )

    # Set learning rate decay.
    # We use a simple exponential decay.
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.num_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.num_iters)
    pbar = tqdm(
        range(args.num_iters), miniters=args.progress_refresh_rate, file=sys.stdout
    )

    # This scripts supports two different renderer_types, "rays" and "image".
    # If `renderer_types == "rays"`,
    #   it renders a number of `args.n_rays` rays like conventional NeRFs  while using significantly less memories.
    # If `renderer_types == "image"`,
    #   it renders a whole image, which allows using LPIPS loss to optimize.

    if (
        args.renderer_type == "rays"
    ):  # calculate total number of rays when `renderer_types == "rays"`
        total_ray_num = dset.rays.origins.shape[0]  # the total number of rays
        batch_begin = 0
        dset.shuffle_rays()
    # Initialize LPIPS function when `renderer_types == "image"` while `args.lpips_loss_weight` > 0.
    if args.lpips_loss_weight > 0 and args.renderer_type == "image":
        lpips_loss_fn = lpips.LPIPS(net="vgg").cuda()

    PSNRs, PSNRs_test = [], [0]
    tv_loss_weight = args.tv_loss_weight
    l1_loss_weight = args.l1_loss_weight

    for iter in pbar:
        optimizer.zero_grad()

        # Render a whole image.
        if args.renderer_type == "image":
            img_index = random.randint(0, dset.n_images - 1)
            image_size = dset.get_image_size(img_index)
            near_t = torch.ones(image_size, device=device).view(-1) * args.near
            far_t = torch.ones(image_size, device=device).view(-1) * args.far
            ray_size = far_t.shape[0]
            rays = Rays(
                directions=dset.rays.dirs[img_index].to(device),
                origins=dset.rays.origins[img_index].to(device),
                grid_idx=torch.zeros(ray_size, device=device, dtype=torch.long),
                near=near_t,
                far=far_t,
            )

            # Render
            (
                ray_length_render,
                alpha_mask,
                feature_render,
            ) = render_module(rays=rays, feature_grid=grid, scaffold=scaffold)

            # MSE Loss
            loss = torch.nn.functional.mse_loss(
                feature_render, dset.rays.gt[img_index].to(device)
            )
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            # LPIPS Loss
            if args.lpips_loss_weight > 0:
                lpips_loss = lpips_loss_fn(
                    feature_render.view(image_size + (3,))
                    .permute(2, 0, 1)
                    .unsqueeze(0),
                    dset.rays.gt[img_index]
                    .view(image_size + (3,))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .cuda(),
                ).sum()
                loss += args.lpips_loss_weight * lpips_loss
        # Render a set of rays
        elif args.renderer_type == "rays":
            batch_end = min(batch_begin + args.n_rays, total_ray_num)
            rgb_gt = dset.rays.gt[batch_begin:batch_end].to(device)
            near_t = torch.ones(batch_end - batch_begin, device=device) * args.near
            far_t = torch.ones(batch_end - batch_begin, device=device) * args.far
            ray_size = far_t.shape[0]
            rays = Rays(
                directions=dset.rays.dirs[batch_begin:batch_end].to(device),
                origins=dset.rays.origins[batch_begin:batch_end].to(device),
                grid_idx=torch.zeros(ray_size, device=device, dtype=torch.long),
                near=near_t,
                far=far_t,
            )

            # Render
            (
                ray_length_render,
                alpha_mask,
                feature_render,
            ) = render_module(rays=rays, feature_grid=grid, scaffold=scaffold)

            loss = torch.nn.functional.mse_loss(feature_render, rgb_gt)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))
            batch_begin = batch_end
            # update batch_begin
            if batch_end >= total_ray_num:
                batch_begin = 0
                dset.shuffle_rays()
        else:
            raise NotImplementedError("no such renderer type")

        # TV_Loss for `grids`
        if args.tv_loss_weight > 0.0:
            tv_loss = grid_TV_loss(grid)
            loss += tv_loss_weight * tv_loss
            tv_loss_weight = tv_loss_weight * lr_factor
        # L1_Loss for `grids`
        if args.l1_loss_weight > 0.0:
            l1_loss = grid_L1_loss(grid)
            loss += l1_loss_weight * l1_loss
            l1_loss_weight = l1_loss_weight * lr_factor
        loss.backward()
        optimizer.step()

        # update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor

        # start evaluation and visialization
        if iter % args.eval_rate == 0 or iter == args.num_iters - 1:
            PSNRs_test = evaluation(
                dset_test,
                device,
                5,
                args,
                render_module,
                grid,
                scaffold,
                logging_path,
                model_path,
                iter,
            )

        # Print the current values of the losses.
        if iter % args.progress_refresh_rate == 0:
            pbar.set_description(
                f"Iteration {iter:05d}:"
                + f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                + f" mse = {loss:.6f}"
            )
            PSNRs = []

        # Upsample grids.
        if iter in args.upsample_step:
            grid = grid_up_sample(grid, upsample_factor=2.0)

            # double num samples
            render_module.num_samples = render_module.num_samples * 2
            render_module.num_samples_inf = render_module.num_samples_inf * 2

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iter / args.num_iters)

            # Re-set optimizer.
            optimizer = torch.optim.Adam(
                [
                    {"params": grid, "lr": args.lr_grids * lr_scale},
                    {
                        "params": render_module.parameters(),
                        "lr": args.lr_nn * lr_scale,
                    },
                ],
                lr=args.lr_nn * lr_scale,
                betas=(args.beta1, args.beta2),
            )
            print(f"Upsample Grids!")

        # Calculate scaffold if `use_scaffold` is True.
        if use_scaffold and iter in args.update_scaffold_step:
            scaffold = render_module.calculate_scaffold(
                feature_grid=grid,
                scaffold_size=[
                    grid[0].shape[0],
                    args.scaffold_size,
                    args.scaffold_size,
                    args.scaffold_size,
                ],
                device=device,
            )
            print("get new scaffold!")

    # Evaluate final results
    evaluation(
        dset_test,
        device,
        1000,
        args,
        render_module,
        grid,
        scaffold,
        logging_path,
        model_path,
        "final",
        True,
        True,
    )


if __name__ == "__main__":
    args = get_argparse()

    main(args)
