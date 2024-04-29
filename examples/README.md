# Lightplane Examples

The [`examples`](./) folder showcases uses of Lightplane Renderer and Splatter.

## Jupyter Notebooks

### Getting started
[`example_1_render_splatter.ipynb` ](./example_1_renderer_splatter.ipynb) demonstrates: 
1) How to set up a simple Renderer object and render a voxel grid representing a randomly-colored 3D sphere.
2) How to set up a Splatter and unproject random image features to a triplane.

### Simple single-scene reconstruction
[`example_2_fit_rendered_mesh.ipynb` ](./example_2_fit_rendered_mesh.ipynb) fits a triplane or a voxel grid given a set of posed RGB images of a cow mesh. The example requires PyTorch3D installed.

## Single-scene reconstruction
[`fit_single_scene`](./fit_single_scene.py) contains a more-advanced training loop implementing fitting of a triplane or a voxel grid given a set of posed RGB images.

#### Example run:
```bash
cd ${LIGHTPLANE_ROOT}/examples/
bash data_download.sh

python ./fit_single_scene.py --config config/synthetic_overfit.json
```

### Supported datasets

The example provides scripts to download and data-load existing datasets. A specific dataset can be selected by setting the `dataset_type` argument to one of:
- `"nerf"`: NeRF dataset
- `"llff"`: LLFF dataset
- `"nsvf"`: NSVF dataset
- `"co3d"`: CO3Dv2 dataset
- `"auto"`: Attempts to automatically infer the dataset type based on the `--datadir` argument.

The [`data_download.sh`](./data_download.sh) script can be used to download some of the latter datasets: "nerf", "llff".

Please refer to [config_util.py](./utils/util/config_util.py) for the full list of configuration arguments.
