# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import typing
import torch
import pytorch3d.io
import pytorch3d.renderer


class Pytorch3DCowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_size: int,
        num_angles: int = 64,
        num_elevations: int = 5,
        device: torch.device = torch.device("cuda:0"),
        bg_depth: float = 5.5,
    ):
        self.image_size = image_size
        self.num_angles = num_angles
        self.num_elevations = num_elevations
        self.bg_depth = bg_depth
        self._render_dataset(device)

    @torch.no_grad()
    def _render_dataset(self, device: torch.device = torch.device("cuda:0")):
        # Load obj file
        obj_filename = os.path.join(
            os.path.dirname(pytorch3d.__file__),
            "../docs/tutorials/data/cow_mesh/cow.obj",
        )
        mesh = pytorch3d.io.load_objs_as_meshes([obj_filename], device=device)
        mesh.offset_verts_(-mesh.verts_list()[0].median(0)[0])
        mesh.scale_verts_(
            torch.tensor([0.9 / v.abs().max() for v in mesh.verts_list()])
        )

        # Rasterization setting
        raster_settings = pytorch3d.renderer.RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Render views into depth maps
        R = []
        T = []
        depths = []
        images = []
        for elevation in torch.linspace(-45, 45, self.num_elevations + 1)[
            : self.num_elevations
        ]:
            for angle in torch.linspace(0, 360, self.num_angles + 1)[: self.num_angles]:
                R_, T_ = pytorch3d.renderer.look_at_view_transform(
                    2.5, elevation, angle, device=device
                )
                R.append(R_)
                T.append(T_)

                camera = pytorch3d.renderer.FoVPerspectiveCameras(
                    R=R_, T=T_, device=device
                )

                rasterizer = pytorch3d.renderer.MeshRasterizer(
                    cameras=camera,
                    raster_settings=raster_settings,
                )
                # result = rasterizer(mesh)

                renderer = pytorch3d.renderer.MeshRendererWithFragments(
                    rasterizer=rasterizer,
                    shader=pytorch3d.renderer.SoftPhongShader(
                        device=device,
                        cameras=camera,
                        lights=pytorch3d.renderer.PointLights(
                            device=device, location=[[0.0, 0.0, -3.0]]
                        ),
                    ),
                )

                image, result = renderer(mesh)

                depth = result.zbuf[0, :, :, 0]
                depths.append(depth)
                images.append(image[0, :, :, :3])

        self.R = torch.cat(R, dim=0).cpu().detach()
        self.T = torch.cat(T, dim=0).cpu().detach()
        self.depths = torch.stack(depths, dim=0).cpu().detach()
        self.images = torch.stack(images, dim=0).cpu().detach()
        self.masks = self.depths < 0
        self.depths[self.masks] = self.bg_depth

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, idx: int):
        return {
            k: getattr(self, k)[idx]
            for k in [
                "images",
                "depths",
                "masks",
                "R",
                "T",
            ]
        }
