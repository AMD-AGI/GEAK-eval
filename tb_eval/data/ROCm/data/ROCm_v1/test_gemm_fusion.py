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


######################################## Imports#######################################
import pytest
import torch

import triton
import triton.language as tl

######################################## Imports#######################################



@triton.jit
def gemm_fusion_kernel(A, B, C, E,  #
                       M, N, K,  #
                       stride_am, stride_ak, stride_bn, stride_bk, stride_cn, stride_ck, stride_em, stride_ek,  #
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)

    a_tile_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak), offsets=(pid * BLOCK_M, 0),
                                   block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_tile_ptr = tl.make_block_ptr(base=B, shape=(N, K), strides=(stride_bn, stride_bk), offsets=(0, 0),
                                   block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))
    c_tile_ptr = tl.make_block_ptr(base=C, shape=(N, K), strides=(stride_cn, stride_ck), offsets=(0, 0),
                                   block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))
    e_tile_ptr = tl.make_block_ptr(base=E, shape=(M, K), strides=(stride_em, stride_ek), offsets=(pid * BLOCK_M, 0),
                                   block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))

    acc_e = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    a = tl.load(a_tile_ptr)
    for i in range(0, N, BLOCK_N):
        b = tl.load(b_tile_ptr)
        o_ab = tl.dot(a, tl.trans(b))
        c = tl.load(c_tile_ptr)
        o_ab = o_ab.to(tl.float16)
        acc_e += tl.dot(o_ab, c)
        b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_N, 0])
        c_tile_ptr = tl.advance(c_tile_ptr, [BLOCK_N, 0])

    acc_e = acc_e.to(tl.float16)
    tl.store(e_tile_ptr, acc_e)

