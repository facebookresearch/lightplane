# Lightplane Splatter

## Renderer module API

The Lightplane Splatter operator is implemented via following interfaces:
- PyTorch modules: [`LightplaneSplatter`](lightplane.LightplaneSplatter) and [`LigthplaneMLPSplatter`](lightplane.LightplaneMLPSplatter)
- Functional interface: [`lightplane_splatter`](lightplane.lightplane_splatter) and [`lightplane_mlp_splatter`](lightplane.lightplane_mlp_splatter)

Visit [Lightplane Splatter API reference](./lightplane_splatter_api.md) for detailed documentation.


## Ray-marched splatting
Lightplane Splatter marches along input rays and pushes the `encoding` of the ray into an output grid-list `output_grid`. The splatter has two forms:
1) **LightplaneSplatter**: For each point along the ray, splats the ray feature to the output grid-list bilinearly or trilinearly.    
2) **LightplaneMLPSplatter**:  For each point along the ray, MLPSplatter first samples the point feature from the corresponding prior input grid `input_grid`, adds the sampled feature to the `encoding` of the ray, passes the latter through an MLP, and splats the MLP output to the grid-list `output_grid`.

The forward passes of both components are described next.

### LightplaneSplatter
```
ray encoding -> splat -> output_grid
```

1) Sample `N=num_samples` equispaced 3D points `pt_3d_i` between the `near` and `far` ray-lengths:
    ```
    pt_3d_1, ..., pt_3d_N = origin + t * direction, t = linspace(near, far, num_samples)
    ```
2) [Splat](feature_grids.md#feature-grid-splatting) the ray `encoding` feature to the output grid-list `output_grid_unnormalized`, and splat a unit scalar to the grid-list `splat_weight` accumulating the total amount of splat-votes into the output grid.

4) Calculate the final normalized output grid `output_grid` by dividing `output_grid_unnormalized` with `splat_weight`.

### LightplaneMLPSplatter
```
input_grid -> f_i -> f_i + ray encoding -> mlp -> sf_i -> splat -> output_grid
```

1) Sample `N=num_samples` equispaced 3D points `pt_3d_i` between the `near` and `far` ray-lengths:
    ```
    pt_3d_1, ..., pt_3d_N = origin + t * direction, t = linspace(near, far, num_samples)
    ```
2) Sample a grid-list `input_grid` for each `pt_3d_i` yielding a sampled feature `f_i`.

3) Add `f_i` to ray `encoding` and pass through MLP yielding splatted feature `sf_i`

4) [Splat](feature_grids.md#feature-grid-splatting) `f_i` to the output grid-list `output_grid_unnormalized`, and splat a unit scalar to the grid-list `splat_weight` accumulating the total amount of splat-votes into the output grid.

5) Calculate the final normalized output grid `output_grid` by dividing `output_grid_unnormalized` with `splat_weight`.


## Splatter configuration

Similar to Renderer, Splatter supports modeling background via coordinate contraction and sampling distant ray-points in disparity space.
Please refer to the Renderer documentation for the description of:
- [Disparity-space background ray-point sampling](lightplane_renderer.md#disparity-space-background-ray-point-sampling)
- [Coordinate contraction](lightplane_renderer.md#coordinate-contraction)

```{note}
The disparity-space sampling is enabled by setting `num_samples_inf` in [`LightplaneSplatter`](lightplane.LightplaneSplatter) or in [`LightplaneMLPSplatter`](lightplane.LightplaneMLPSplatter) to an integer value > 0.

Coordinate contraction is enabled by setting `contract_coords=True` in [`LightplaneSplatter`](lightplane.LightplaneSplatter) or in [`LightplaneMLPSplatter`](lightplane.LightplaneMLPSplatter).
```