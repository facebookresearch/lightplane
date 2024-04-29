# Lightplane Renderer API

The renderer API consists of two main components:
- **[`LightplaneRenderer`](lightplane.LightplaneRenderer) module** which holds learnable parameters of the mlp decoder and can be reused to render different input grids
- **[`lightplane_renderer`](lightplane.lightplane_renderer) function** providing the lowest-level interface to the renderer.

Visit the [Lightplane Renderer](./lightplane_renderer.md) section for a more detailed overview of the renderer's functionality.

```{eval-rst}
.. autoclass:: lightplane.LightplaneRenderer
    :members:
```

```{eval-rst}
.. autofunction:: lightplane.lightplane_renderer
```
