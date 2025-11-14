# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
"""
Testing the (FP8) case of a dot op that consumes the output (MFMA) of
another dot op as an input.

"""
######################################## Imports#######################################
import math
import pytest
import torch

import triton
import triton.language as tl
######################################## Imports#######################################

########################## HELPER utils ##########################
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8: tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz
########################## HELPER utils ##########################


@triton.jit
def _chained_dot(
    Q,
    K,
    V,
    Out,
    q_desc,
    k_desc,
    v_desc,
    s_sc,
    s_desc,
    o_sc,
    stride_qz,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vd,
    stride_vn,
    stride_oz,
    stride_om,
    stride_od,
    Z,
    M,
    N,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)
    qkv_offset = off_z * stride_qz
    Q_block_ptr = tl.make_block_ptr(base=Q + qkv_offset, shape=(N, BLOCK_D), strides=(stride_qm, stride_qd),
                                    offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qkv_offset, shape=(BLOCK_D, N), strides=(stride_kd, stride_kn),
                                    offsets=(0, 0), block_shape=(BLOCK_D, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qkv_offset, shape=(N, BLOCK_D), strides=(stride_vn, stride_vd),
                                    offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_D), order=(0, 1))

    s_scale = q_desc * k_desc * s_sc
    acc_scale = s_desc * v_desc * o_sc

    q = tl.load(Q_block_ptr)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    lo, hi = 0, N
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(K_block_ptr)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)

        if USE_FP8:
            s *= s_scale

        v = tl.load(V_block_ptr)
        acc += tl.dot(s.to(v.dtype), v)

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    if USE_FP8:
        acc *= acc_scale

    O_block_ptr = tl.make_block_ptr(base=Out + qkv_offset, shape=(N, BLOCK_D), strides=(stride_om, stride_od),
                                    offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

