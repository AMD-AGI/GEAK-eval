# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
######################################## Imports ######################################## 
import itertools
import os
import re

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

######################################## Imports ######################################## 



@triton.jit
def matmul_no_scf_kernel(a_ptr, b_ptr, c_ptr,  #
                         M, N, K,  #
                         stride_am, stride_ak,  #
                         stride_bk, stride_bn,  #
                         stride_cm, stride_cn,  #
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                         FLOAT16_OUTPUT: tl.constexpr, USE_TMA_EPILOGUE: tl.constexpr  #
                         ):
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    c = tl.dot(a, b)

    if FLOAT16_OUTPUT:
        c = c.to(tl.float16)

    if USE_TMA_EPILOGUE:
        c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
        tl.store(c_block_ptr, c)
    else:
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c)

