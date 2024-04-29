# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import math
import torch

from typing import Optional
from .ray_utils import Rays

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError:
    raise ImportError(
        "The `plotly` package is required for visualisation functions."
        " Execute `pip install plotly` to install it."
    )


def visualize_rays_plotly(
    rays: Rays,
    output_path: Optional[str] = None,
    max_display_rays_per_grid: int = -1,
    ncols: Optional[int] = None,
    ray_line_width: float = 1.0,
    ray_point_marker_size: float = 1.0,
    ray_pixel_colors: Optional[torch.Tensor] = None,
):
    """
    Visualizes rays using plotly. The rays are visualized in a grid of subplots,
    where each subplot corresponds to a feature grid. The rays are visualized as
    lines and points, with the points (optionally) colored according to the
    provided pixel colors. The near and far points of the rays are also visualized
    as points. The resulting plotly figure can be saved as an html file using
    the `output_path` argument. The plotly figure can then be interactively explored
    in a web browser.

    Args:
        rays: A `Rays` dataclass to visualize.
        output_path: Optional path to save the plotly figure as an html file.
        max_display_rays_per_grid: Maximum number of rays to display per feature grid.
        ncols: Number of columns in the plot grid.
        ray_line_width: Width of the plotted ray lines.
        ray_point_marker_size: Size of the plotted ray points.
        ray_pixel_colors: Optional tensor of shape `(N, 3)` containing RGB pixel colors.
            If provided, the ray points near the origin will be colored with these
            pixel colors. The pixel colors should be in the range `[0, 1]`.

    Returns:
        fig: The plotly figure.

    """

    num_grids = rays.grid_idx.max().item() + 1

    if ncols is None:
        ncols = int(math.ceil(math.sqrt(num_grids)))
    fig_rows = num_grids // ncols
    if num_grids % ncols != 0:
        fig_rows += 1  # allow for non-uniform rows
    fig_cols = ncols
    fig_type = [{"type": "scene"}]
    specs = [fig_type * fig_cols] * fig_rows
    subplot_titles = [f"scene_{s}" for s in range(num_grids)]
    fig = make_subplots(
        rows=fig_rows,
        cols=fig_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[1.0] * fig_cols,
    )

    for grid_idx, title in enumerate(subplot_titles):
        rays_grid = rays[rays.grid_idx == grid_idx]
        row, col = grid_idx // ncols, grid_idx % ncols
        if max_display_rays_per_grid > 0:
            rays_grid = rays_grid[:max_display_rays_per_grid]
        _add_rays_trace(
            fig,
            rays_grid,
            grid_idx,
            row,
            col,
            title,
            ray_line_width,
            ray_point_marker_size,
            ray_pixel_colors,
        )

    if output_path is not None:
        fig.write_html(output_path)

    return fig


def _add_rays_trace(
    fig,
    rays: Rays,
    subplot_idx: int,
    row: int,
    col: int,
    trace_name: str,
    line_width: float,
    marker_size: float,
    ray_pixel_colors: torch.Tensor | None,
):
    """
    Add a trace to the plot for visualizing rays.

    Note: This function is strongly inspired by PyTorch3D's `plot_scene` function.

    Args:
        fig : The figure object to add the trace to.
        rays: The rays to visualize.
        subplot_idx: The index of the subplot to add the trace to.
        row: The row index of the subplot.
        col: The column index of the subplot.
        trace_name: The name of the trace.
        line_width: The width of the ray lines.
        marker_size: The size of the ray points.
        ray_pixel_colors: The pixel colors associated with the rays.
            If provided, the ray points will be colored accordingly.
    """

    # ndc box lines
    one_line_cube = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 1),
        (1, 1, 0),
        (0, 1, 0),
        (0, 1, 1),
    ]

    one_line_cube_ndc = (
        2
        * torch.tensor(
            one_line_cube,
            dtype=torch.float32,
            device=rays.device,
        )
        - 1
    )
    fig.add_trace(
        go.Scatter3d(
            x=one_line_cube_ndc[:, 0].detach().cpu().numpy().astype(float),
            y=one_line_cube_ndc[:, 1].detach().cpu().numpy().astype(float),
            z=one_line_cube_ndc[:, 2].detach().cpu().numpy().astype(float),
            marker={"size": 0.1},
            line={"width": line_width},
            name=trace_name + "_volume_bounds",
        ),
        row=row + 1,
        col=col + 1,
    )

    # ray line endpoints
    ray_lines_endpoints = torch.stack(
        [rays.origins + rays.directions * x[:, None] for x in [rays.near, rays.far]],
        dim=1,
    )

    # make the ray lines for plotly plotting
    nan_tensor = torch.tensor(
        [[float("NaN")] * 3],
        device=ray_lines_endpoints.device,
        dtype=ray_lines_endpoints.dtype,
    )

    ray_lines = torch.empty(size=(1, 3), device=ray_lines_endpoints.device)

    for ray_line in ray_lines_endpoints:
        # We combine the ray lines into a single tensor to plot them in a
        # single trace. The NaNs are inserted between sets of ray lines
        # so that the lines drawn by Plotly are not drawn between
        # lines that belong to different rays.
        ray_lines = torch.cat((ray_lines, nan_tensor, ray_line))
    x, y, z = ray_lines.detach().cpu().numpy().T.astype(float)

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            marker={"size": 0.1},
            line={"width": line_width},
            name=trace_name + "_rays",
        ),
        row=row + 1,
        col=col + 1,
    )

    # plot the ray points
    for is_far in [False, True]:
        near_or_far = ray_lines_endpoints[:, int(is_far)]
        ray_points = near_or_far.view(-1, 3).detach().cpu().numpy().astype(float)

        marker_settings = {"size": marker_size}
        if not is_far and ray_pixel_colors is not None:
            assert ray_pixel_colors.shape[0] == ray_points.shape[0]
            assert ray_pixel_colors.shape[1] == 3
            rgb = (ray_pixel_colors.clamp(0.0, 1.0) * 255).int()
            template = "rgb(%d, %d, %d)"
            color = [template % (r, g, b) for r, g, b in rgb]
            marker_settings["color"] = color

        fig.add_trace(
            go.Scatter3d(
                x=ray_points[:, 0],
                y=ray_points[:, 1],
                z=ray_points[:, 2],
                mode="markers",
                name=trace_name + f"_points_{'far' if is_far else 'near'}",
                marker=marker_settings,
            ),
            row=row + 1,
            col=col + 1,
        )

    # Access the current subplot's scene configuration
    plot_scene = "scene" + str(subplot_idx + 1)
    current_layout = fig["layout"][plot_scene]

    # update the bounds of the axes for the current trace
    all_ray_points = ray_lines_endpoints.reshape(-1, 3)
    ray_points_center = all_ray_points.mean(dim=0)
    max_expand = (all_ray_points.max(0)[0] - all_ray_points.min(0)[0]).max().item()
    _update_axes_bounds(ray_points_center, float(max_expand), current_layout)


def _update_axes_bounds(
    verts_center: torch.Tensor,
    max_expand: float,
    current_layout: "Scene",  # pyre-ignore[11]
) -> None:  # pragma: no cover
    """
    Takes in the vertices' center point and max spread, and the current plotly figure
    layout and updates the layout to have bounds that include all traces for that subplot.
    Args:
        verts_center: tensor of size (3) corresponding to a trace's vertices' center point.
        max_expand: the maximum spread in any dimension of the trace's vertices.
        current_layout: the plotly figure layout scene corresponding to the referenced trace.
    """
    verts_center = verts_center.detach().cpu()
    verts_min = verts_center - max_expand
    verts_max = verts_center + max_expand
    bounds = torch.t(torch.stack((verts_min, verts_max)))

    # Ensure that within a subplot, the bounds capture all traces
    old_xrange, old_yrange, old_zrange = (
        current_layout["xaxis"]["range"],
        current_layout["yaxis"]["range"],
        current_layout["zaxis"]["range"],
    )
    x_range, y_range, z_range = bounds
    if old_xrange is not None:
        x_range[0] = min(x_range[0], old_xrange[0])
        x_range[1] = max(x_range[1], old_xrange[1])
    if old_yrange is not None:
        y_range[0] = min(y_range[0], old_yrange[0])
        y_range[1] = max(y_range[1], old_yrange[1])
    if old_zrange is not None:
        z_range[0] = min(z_range[0], old_zrange[0])
        z_range[1] = max(z_range[1], old_zrange[1])

    xaxis = {"range": x_range}
    yaxis = {"range": y_range}
    zaxis = {"range": z_range}
    current_layout.update({"xaxis": xaxis, "yaxis": yaxis, "zaxis": zaxis})
