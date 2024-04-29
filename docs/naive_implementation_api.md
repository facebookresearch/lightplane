# Naive Implementation API

Together with highly-optimized renderer and splatter implementations, we also provide naive auto-grad implementations of the renderer and splatter.

These implementations are numerically equivalent to their optimized versions, which is used for unit-testing the correctness of both forward and backward passes of the optimized versions.


```{eval-rst}
.. autofunction:: lightplane.lightplane_renderer_naive

.. autofunction:: lightplane.lightplane_splatter_naive

.. autofunction:: lightplane.lightplane_mlp_splatter_naive
```