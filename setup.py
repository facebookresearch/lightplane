# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python

from setuptools import find_packages, setup


def setup_package():
    setup(
        name="lightplane",
        version="0.1",
        description="Lightplane",
        packages=find_packages(exclude=["tests", "examples", "scratch"]),
        install_requires=[
            "cogapp",
            "triton==2.1.0",
            "configargparse",
            "tqdm"
        ],
    )


if __name__ == "__main__":
    setup_package()
