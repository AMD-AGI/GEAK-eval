# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import numpy as np
import pytest
import torch

import triton
import triton.language as tl
######################################## Imports ######################################## 

#####################################
# Triton Kernels for randint
#####################################

BLOCK: tl.constexpr = 1024

@triton.jit
def randn_kernel_runtime_seed(X, N, seed, dtype: tl.constexpr):
    pid = tl.program_id(0).to(dtype)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    rand = tl.rand(seed, offset)
    tl.store(X + offset, rand, mask=offset < N)

@triton.jit
def randn_kernel_const_seed(X, N, seed: tl.constexpr, dtype: tl.constexpr):
    pid = tl.program_id(0).to(dtype)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    rand = tl.rand(seed, offset)
    tl.store(X + offset, rand, mask=offset < N)

