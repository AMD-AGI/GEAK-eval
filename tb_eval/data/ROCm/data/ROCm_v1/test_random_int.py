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
def randint_kernel_runtime_seed(X, N, seed_val): # Kernel for runtime seed
    pid = tl.program_id(0).to(X.dtype.element_ty) # pid uses X's dtype for consistency if X is int
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    rand = tl.randint(seed_val, offset)
    tl.store(X + offset, rand, mask=offset < N)

@triton.jit
def randint_kernel_const_seed(X, N, seed_val: tl.constexpr): # Kernel for const seed
    pid = tl.program_id(0).to(X.dtype.element_ty) # pid uses X's dtype for consistency
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    rand = tl.randint(seed_val, offset)
    tl.store(X + offset, rand, mask=offset < N)

