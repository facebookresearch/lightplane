# 3D feature grids

Lightplane Renderer or Splatter represent 3D scenes as a grid-list `grid` of `M` feature grids `g_i`:
```python
grid: List[torch.Tensor] = [g_1, ..., g_M]
```
Each `g_i` is a 5D tensor of shape `(B, D, H, W, C)`, with batch size `B` , number of feature channels `C` (equal for all grids), and the grid depth, height, and width `D`, `H`, and `W` respectively.

**Representing Voxel grid**: If all `g_i`'s dimensions are non-singular (`H>1 or D>1 or W>1`), `g_i` represents a batch of voxel grids with `C`-dimensional features.

**Representing 2D plane**: In case one of the `g_i`'s dimensions is singular (`H == 1 or D==1 or W==1`), `g_i` is treated as a planar 2D grid whose cells represent all points along a 3D line passing through the cell's center orthogonal to the plane (akin to a triplane).

The grid-list is a mandatory argument for both [`LightplaneRenderer`](lightplane.LightplaneRenderer) and [`LigthplaneMLPSplatter`](lightplane.LightplaneMLPSplatter), and is output by [`LigthplaneSplatter`](lightplane.LightplaneSplatter) and [`LigthplaneMLPSplatter`](lightplane.LightplaneMLPSplatter).


````{note}
Below are examples of representing the latest grid-based representations:

**Triplane**:
```
grid = [
    torch.randn(B, 1, H, W, C),  # xy plane
    torch.randn(B, D, 1, W, C),  # xz plane
    torch.randn(B, D, H, 1, C),  # yz plane
]
```

**Voxel grid**:
```
grid = [torch.randn(B, D, H, W, C)]
```

**Voxel grid and triplane**:
```
grid = [
    torch.randn(B, D_voxels, H_voxels, W_voxels, C),  # voxel grid
    torch.randn(B, 1, H, W, C),  # xy plane
    torch.randn(B, D, 1, W, C),  # xz plane
    torch.randn(B, D, H, 1, C),  # yz plane
]
```
````

```{note}
All grids in a grid-list have to share the same number of channels `C` and batch size `B`, however, the spatial dimensions `D`, `H`, `W` of each grid can be arbitrarily different across grids.
```


## Grid coordinate frame
We follow the [PyTorch grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch-nn-functional-grid-sample) coordinate convention.

Each grid `g` represents a 3D cube of side `[-1, 1]` centered at `[0, 0, 0]`. All rendering rays thus have to be correctly emitted to render points inside the latter cube.

A continuous 3D point `pt_3d=[x,y,z]` with coordinates `[-1.0, -1.0, -1.0]` indexes the outer corner of `g`'s voxel at integer cell `[0, 0, 0]` in the 3D grid, while a 3D point `[1.0, 1.0, 1.0]` corresponds to the opposite cell at `[D-1, H-1, W-1]`.

A 3D point `pt_3d=[x,y,z]` indexes the `g` tensor in reverse order of `p`'s dimensions, i.e.:
- `x=pt_3d[0]` indexes the width (i.e. 3rd `W`-dimension) of `g`
- `y=pt_3d[1]` indexes the height (i.e. 2nd `H`-dimension) of `g`
- `z=pt_3d[2]` indexes the depth (i.e. 1st `D`-dimension) of `g`

For example, in a 3D grid `g` of size `[2, 4, 8]`, a continuous 3D point `pt_3d=[-0.25, 0.25, 0.0]` corresponds to `g`'s element at integer coordinates `[0, 1, 3]`.


## Feature grid sampling
Lightplane Renderer and Splatter rely on a function `sample_grid(pt_3d, grid)` that sums the `C`-dimensional vectors `sample(pt_3d, g)` sampled at a continuous 3D location `pt_3d=[x,y,z]` from each grid `g` of the grid-list `grid`:
```python
sample_grid(pt_3d, grid) = sum(sample(pt_3d, g) for g in grid)
```
Here, `sample` interpolates the grid `g` using a linearly-weighted scheme depending on the number of `g`'s non-singular dimensions:
- **3 non-singular dimensions**: If all spatial dimensions of the grid `g` are non-singular (i.e. `D>1 and H>1 and W>1`), the feature is *trilinearly interpolated* from the grid at location `pt_3d`.
    ```python
    # define a random voxel grid
    B, D, H, W, C = 4, 3, 7, 5, 6
    g = torch.randn(B, D, H, W, C)
    # sample with trilinear interpolation
    sampled_feature = trilinear_interpolation(g, pt_3d)
    ```
- **2 non-singular dimensions**: If one of the spatial dimensions of the grid is singular (i.e. `D==1 or H==1 or W==1`), the feature is *bilinearly interpolated* from the 2D grid `g_2d` (obtained by squeezing the singular dimension of `g`) at the 2D coordinate `pt_2d` comprising the 2 coordinates of the 3D point `pt_3d` corresponding to non-singular dimensions. The following example demonstrates Lightplane's bilinear interpolation of a `(B, D, 1, W, C)` grid.
    ```python
    # define a random grid with a singular height
    B, D, H, W, C = 4, 3, 1, 5, 6
    g = torch.randn(B, D, H, W, C)
    # squeeze the singular height-dimension to obtain the corresponding 2d grid
    g_2d = g.squeeze(2)
    # the height-dimension is singular so we only sample the x- and z-coordinates
    # using bilinear interpolation
    x, y, z = pt_3d
    sampled_feature = bilinear_interpolation(g, [x, z])
    ```


## Feature grid splatting
Lightplane Splatter pushes 3d-point-specific `C`-dimensional vectors `f(pt_3d)` to each grid `g` of the grid-list `grid` by means of [splatting](https://www2.cs.uh.edu/~chengu/Teaching/Fall2023/Lectures/Lec9_DVR-Splatting.pdf):
```python
grid_splatted = [splat_to_grid(f(pt_3d), g_i) for g_i in grid]
```
Similar to the renderer, splatting is either bilinear or trilinear depending on the number of singular dimensions of `g_i`.
- **3 non-singular dimensions**: If all spatial dimensions of the grid `g` are non-singular (i.e. `D>1 and H>1 and W>1`), the feature is *trilinearly splatted* to the grid at location `pt_3d`.
    ```python
    # define a random voxel grid
    B, D, H, W, C = 4, 3, 7, 5, 6
    g = torch.randn(B, D, H, W, C)
    # splat feature f(pt_3d) to the grid g to create a new grid
    g_splatted = trilinear_splatting(g, pt_3d, f(pt_3d))
    ```
- **2 non-singular dimensions**: If one of the spatial dimensions of the grid is singular (i.e. `D==1 or H==1 or W==1`), the feature is *bilinearly splatted* to the 2D grid `g_2d` (obtained by squeezing the singular dimension of `g`) at the 2D coordinate `pt_2d` comprising the 2 coordinates of the 3D point `pt_3d` corresponding to non-singular dimensions. The following example demonstrates Lightplane's bilinear splatting of a `(B, D, 1, W, C)` grid.
    ```python
    # define a random grid with a singular height
    B, D, H, W, C = 4, 3, 1, 5, 6
    g = torch.randn(B, D, H, W, C)
    # squeeze the singular height-dimension to obtain the corresponding 2d grid
    g_2d = g.squeeze(2)
    # the height-dimension is singular so we only sample the x- and z-coordinates
    # using bilinear interpolation
    x, y, z = pt_3d
    g_splatted = bilinear_splatting(g, [x, z], f(pt_3d))
    ```