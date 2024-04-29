# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Any, Optional, Tuple, Union

import cv2
import imageio
import numpy as np
import torch

# def save_img(file_path:str, image: np.array) -> None:
#     imageio.imwrite(file_path, image)


def save_image_list_to_video(video_filename, image_list, fps=20):
    frame_height, frame_width = image_list[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID' if you want an AVI file

    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
    for frame in image_list:
        # OpenCV expects colors in BGR format, convert if your images are RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    # Release the video writer
    out.release()


def convert_depth_image_to_colormap(
    depth_img: torch.Tensor,
    max_depth: Optional[float] = None,
    min_depth: Optional[float] = None,
    max_quantile: Optional[float] = 0.95,
    min_quantile: Optional[float] = 0.05,
    colormap=None,
):
    """Convert depth image to a colormap.
    Args:
        depth_img: (H, W)
        max_depth: None or scalar
    """
    if max_depth is None:
        max_depth = torch.quantile(depth_img, max_quantile).item()
    if min_depth is None:
        min_depth = torch.quantile(depth_img, min_quantile).item()

    depth_img = torch.clamp(depth_img, min=min_depth, max=max_depth)
    depth_img = (depth_img - min_depth) / (max_depth - min_depth)

    if depth_img.is_cuda:
        depth_img = depth_img.detach().cpu().numpy()
    else:
        depth_img = depth_img.numpy()

    depth_img = depth_img * 255
    depth_img = np.uint8(depth_img)
    if colormap is None:
        colomap = cv2.COLORMAP_MAGMA
    depth_img = cv2.applyColorMap(depth_img, colormap)
    depth_img[np.isinf(depth_img)] = 0
    return depth_img


def safe_create_dir(dir_path: str) -> None:
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)


def get_save_path(save_dir: str) -> Tuple[str, str]:
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    logging_path = os.path.join(save_dir, "logging")
    if os.path.exists(logging_path) is False:
        os.makedirs(logging_path)

    model_path = os.path.join(save_dir, "model.pt")

    return logging_path, model_path
