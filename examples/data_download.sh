# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

python -m pip install gdown

mkdir -p data
cd data

# Download llff
gdown 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g
unzip nerf_llff_data.zip && mv nerf_llff_data llff
rm nerf_llff_data.zip

# Download real night lego
gdown 1PG-KllCv4vSRPO7n5lpBjyTjlUyT8Nag
tar -xvf lego_real_night_radial.tar.gz && mkdir -p custom  && mv lego_real_night_radial custom/lego
rm lego_real_night_radial.tar.gz

# Download NeRF synthetic
gdown 1A_zU6Eu-qy4XhtNkBLeATYFieLms3bvp
unzip nerf_synthetic.zip
rm _MACOSX && rm nerf_synthetic.zip

cd ..
