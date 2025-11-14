# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl

######################################## Imports ######################################## 

@triton.jit
def batched_vecmat(
        # inputs
        A,  # shape: [dim_m, dim_k]
        B,  # shape: [dim_m, dim_n, dim_k]
        # dimensions
    dim_m, dim_n, dim_k,
        # outputs
        output,
        # block information
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)
    # Output tile
    output_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_n \
        + (n_index * block_n + tl.arange(0, block_n))[None, :]

    vecmat = tl.zeros([block_m, block_n], dtype=A.dtype.element_ty)
    k_blocks = dim_k // block_k
    for k_index in range(k_blocks):
        # Load A tile
        a_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, :]
        a = tl.load(A + a_tile)

        # Load B tile, transposed to [n, m, k] in order to broadcast A on a
        # leading dimension.
        b_tile = (m_index * block_m + tl.arange(0, block_m))[None, :, None] * dim_n * dim_k \
            + (n_index * block_n + tl.arange(0, block_n))[:, None, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, None, :]
        b = tl.load(B + b_tile)

        expanded_a, _ = tl.broadcast(a, b)
        vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))

    tl.store(output + output_tile, vecmat)


