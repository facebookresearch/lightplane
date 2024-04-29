# Rendering rays
Rendering rays are representend using Lightplane's [`Rays`](lightplane.Rays) class.

Lightplane rays are parametrized with two 3D vectors `origin` and `direction`. A ray-point `pt_3d(t)` at a scalar ray-length `t` is defined as:
```
pt_3d(t) = origin + t * direction
```
Note that `direction` does not have to be `l2`-normalized.

For raymarching, the `Rays` class further defines scalars `near` and `far` comprising the minimum and maximum rendering ray-length `t`.

Finally, rays can carry a high-dimensional encoding utilizable by both Renderer and Splatter, stored in the optional `Rays.encoding` field.
