# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ########################################
import pytest
import torch
import triton
import triton.language as tl
# numpy is not strictly needed here if output is directly compared
######################################## Imports ########################################

@triton.jit
def swizzle2d_kernel(output, size_i, size_j, size_g):
    for i in tl.range(0, size_i, 1):
        for j in tl.range(0, size_j, 1):
            new_i, new_j = tl.swizzle2d(i, j, size_i, size_j, size_g)
            tl.store(output + new_i * size_j + new_j, i * size_j + j)

