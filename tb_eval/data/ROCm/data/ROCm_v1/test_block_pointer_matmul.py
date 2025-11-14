# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import pytest
import torch

import triton
import triton.language as tl
import os
######################################## Imports ######################################## 

@triton.jit
def matmul_no_scf_with_advance_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr  #
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                    block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
    # Below two lines are just for testing negative offsets for the `advance` API, which could be removed
    a_block_ptr = tl.advance(a_block_ptr, (BLOCK_M, -BLOCK_K))
    a_block_ptr = tl.advance(a_block_ptr, (-BLOCK_M, BLOCK_K))
    a = tl.load(a_block_ptr, boundary_check=(1, ), padding_option="zero")
    b = tl.load(b_block_ptr, boundary_check=(0, ), padding_option="zero")

    c = tl.dot(a, b)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)

