# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# A100s cause illegal memory access when this is enabled
ALLOW_TF32 = False

# If False, makes sure that all threads in a warp are synchronized at the end of each function
ALLOW_WARP_DIVERGENCE = False

# minimum size of a block of variables in Triton (=lower bound on BLOCK_SIZE)
MIN_BLOCK_SIZE = 16
