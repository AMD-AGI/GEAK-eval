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
"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""
######################################## Imports ######################################## 

# import numpy as np
import pytest
import torch

import triton
import triton.language as tl
######################################## Imports ######################################## 


@triton.jit
def flash_fwd_kernel(Q, K, V, sm_scale,  #
                L, M,  #
                Out,  #
                stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_kz, stride_kh, stride_kn, stride_kk,  #
                stride_vz, stride_vh, stride_vk, stride_vn,  #
                stride_oz, stride_oh, stride_om, stride_on,  #
                Z, H, N_CTX, D0,  #
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # TODO: may replace with TMA store without range offset
    # initialize offsets for store
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    stride_qh_2d = stride_qh // stride_qm // stride_qk

    q_tile_ptr = tl.make_block_ptr(
        base=Q,
        shape=(D0, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(off_hz * stride_qh_2d + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    k_tile_ptr = tl.make_block_ptr(
        base=K,
        shape=(D0, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(off_hz * stride_qh_2d, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    v_tile_ptr = tl.make_block_ptr(
        base=V,
        shape=(D0, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(off_hz * stride_qh_2d, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    out_tile_ptr = tl.make_block_ptr(
        base=Out,
        shape=(D0, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(off_hz * stride_qh_2d + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # load q: it will stay in SRAM throughout
    q = tl.load(q_tile_ptr)

    # loop over k, v and update accumulators
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_tile_ptr, boundary_check=(0, 1))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(tl.float16)
        v = tl.load(v_tile_ptr, boundary_check=(0, 1))
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_tile_ptr = tl.advance(k_tile_ptr, [BLOCK_N, 0])
        v_tile_ptr = tl.advance(v_tile_ptr, [BLOCK_N, 0])
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_prev)
    tl.store(m_ptrs, m_prev)

    acc = acc.to(tl.float16)
    tl.store(out_tile_ptr, acc, boundary_check=(0, 1))

