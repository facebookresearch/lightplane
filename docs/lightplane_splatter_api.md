# Lightplane Splatter API

The renderer API consists of two main components:
- **[`LightplaneSplatter`](lightplane.LightplaneSplatter) module** which can be reused to splat features to different output grids given input rays.
- **[`LightplaneMLPSplatter`](lightplane.LightplaneMLPSplatter) module** which holds learnable parameters of the splatting mlp and which can be reused to splat features to different output grids given input rays and input grids.
- **[`lightplane_splatter`](lightplane.lightplane_splatter), [`lightplane_mlp_splatter`](lightplane.lightplane_mlp_splatter) functions** providing the lowest-level interface to the splatter and the mlp-based splatter respectively.

Visit the [Lightplane Splatter](./lightplane_splatter.md) section for a more detailed overview of the splatter's functionality.

```{eval-rst}
.. autoclass:: lightplane.LightplaneSplatter
    :members:
```

```{eval-rst}
.. autoclass:: lightplane.LightplaneMLPSplatter
    :members:
```

```{eval-rst}
.. autofunction:: lightplane.lightplane_splatter
```

```{eval-rst}
.. autofunction:: lightplane.lightplane_mlp_splatter
```
