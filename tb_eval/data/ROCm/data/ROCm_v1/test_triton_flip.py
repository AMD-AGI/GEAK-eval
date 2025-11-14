# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ########################################
import pytest
import torch
import triton
import triton.language as tl
import numpy as np
######################################## Imports ########################################

@triton.jit
def flip_kernel(X, Z, N: tl.constexpr, M: tl.constexpr):
    offx = tl.arange(0, M)
    offy = tl.arange(0, N) * M
    off2d = offx[None, :] + offy[:, None]
    x = tl.load(X + off2d)
    x = tl.flip(x)
    tl.store(Z + off2d, x)


