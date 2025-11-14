# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import pytest
import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
######################################## Imports ######################################## 



@triton.jit
def matmul_kernel(  #
        a_ptr, scale_ptr, b_ptr, output_ptr,  #
        M, N, K_MXFP,  # K_MXFP is the number of mxfp vectors in a row of a. Otherwise it's just K
        stride_am, stride_ak,  #
        stride_sm, stride_sk,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr, a_type: tl.constexpr, b_type: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    IS_SCALED: tl.constexpr = a_type is not None and b_type is not None
    DIV_FACTOR: tl.constexpr = 2 if IS_SCALED and a_type == "e2m1" else 1
    # We pass K_MXFP to make explicit that KB is multiple of 32 and KA is multiple of 16 or 32
    # for the pipeliner divisibility condition
    KA = K_MXFP if not IS_SCALED else K_MXFP * (32 // DIV_FACTOR)
    KB = K_MXFP if not IS_SCALED else K_MXFP * 32
    BLOCK_AK: tl.constexpr = BLOCK_K // DIV_FACTOR
    offs_k = tl.arange(0, BLOCK_K)
    offs_ak = tl.arange(0, BLOCK_AK)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if IS_SCALED:
        BLOCK_SK: tl.constexpr = BLOCK_K // 32
        offs_sk = tl.arange(0, BLOCK_SK)
        scale_ptrs = scale_ptr + (offs_am[:, None] * stride_sm + offs_sk[None, :] * stride_sk)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(KB, BLOCK_K), num_stages=NUM_STAGES):
        mask_a = (offs_am[:, None] < M) & (offs_ak[None, :] + k * BLOCK_AK < KA)
        mask_b = ((offs_k[:, None] + k * BLOCK_K) < KB) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        if IS_SCALED:
            # Adapted scale indexing and dot_scaled operation
            mask_scale = (offs_am[:, None] < M) & (offs_sk[None, :] + k * BLOCK_SK < K_MXFP)
            a_scale = tl.load(scale_ptrs, mask=mask_scale, other=0)
            accumulator = tl.dot_scaled(a, a_scale, a_type, b, None, b_type, acc=accumulator)
        else:
            accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_AK * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        if IS_SCALED:
            scale_ptrs += BLOCK_SK * stride_sk
    OUT_DTYPE = tl.bfloat16 if IS_SCALED else tl.float16
    accumulator = accumulator.to(OUT_DTYPE)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator, mask=mask_c)


@triton.jit
def mxfp_to_bf16_kernel(
    x_ptr,
    scale_ptr,
    mxfp_ptr,
    N,
    e_bits: tl.constexpr,
    m_bits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # x.shape ==     (N, 32) for fp8 or (N, 16) for fp4
    # scale.shape == (N,)
    # out.shape   == (N, 32)
    is_fp8: tl.constexpr = e_bits + m_bits == 7
    # fp8: BLOCK_SIZE -> BLOCK_SIZE // 32, 32
    # fp4: BLOCK_SIZE // 2 -> BLOCK_SIZE // 32 , 16
    PARALLEL_DIM: tl.constexpr = BLOCK_SIZE // 32
    LAST_DIM: tl.constexpr = 32 if is_fp8 else 16
    LOAD_SIZE: tl.constexpr = LAST_DIM * PARALLEL_DIM

    offsets = (tl.program_id(0) * LOAD_SIZE + tl.arange(0, PARALLEL_DIM)[:, None] * LAST_DIM +
               tl.arange(0, LAST_DIM)[None, :])
    x = tl.load(x_ptr + offsets, mask=offsets < N * LAST_DIM)

    offsets = tl.program_id(0) * PARALLEL_DIM + tl.arange(0, PARALLEL_DIM)[:, None]
    scale = tl.load(scale_ptr + offsets, mask=offsets < N)
    tl.static_assert(scale.dtype == tl.uint8)
    tl.static_assert(x.dtype == tl.uint8)

    scale_bf16 = (scale.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)
    if is_fp8:
        if e_bits == 5 and m_bits == 2:
            x_f8 = x.to(tl.float8e5, bitcast=True)
            x_bf16 = x_f8.to(tl.bfloat16)
            # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
            non_finite_mask: tl.constexpr = ((1 << e_bits) - 1) << m_bits
            non_finite_mask_bf16: tl.constexpr = ((1 << 8) - 1) << 7
            x_bf16 = tl.where(
                x & non_finite_mask == non_finite_mask,
                (x_bf16.to(tl.uint16, bitcast=True) | non_finite_mask_bf16).to(tl.bfloat16, bitcast=True),
                x_bf16,
            )
        else:
            tl.static_assert(e_bits == 4 and m_bits == 3)
            x_f8 = x.to(tl.float8e4nv, bitcast=True)
            x_bf16 = x_f8.to(tl.bfloat16)
    else:
        # e2m1
        em0 = x & 0x70
        em1 = x & 0x7
        x0 = (em0.to(tl.uint16) << 2) | ((x & 0x80).to(tl.uint16) << 8)
        x1 = (em1.to(tl.uint16) << (2 + 4)) | ((x & 0x8).to(tl.uint16) << (8 + 4))
        # Three cases:
        # 1) x is normal and non-zero: Correct bias
        x0 = tl.where((em0 & 0x60) != 0, x0 + ((127 - 1) << 7), x0)
        x1 = tl.where((em1 & 0x6) != 0, x1 + ((127 - 1) << 7), x1)
        # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in bf16
        x0 = tl.where(em0 == 0x10, 16128 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x1, 16128 | (x1 & 0x8000), x1)
        # 3) x is zero, do nothing
        x_bf16 = tl.interleave(x0, x1).to(tl.bfloat16, bitcast=True)
    # Multiplication preserves infs and NaNs in x_bf16
    mxfp = x_bf16 * scale_bf16
    # If scale is NaN, we encode it as an bf16 inf, so we need to correct for that
    mxfp = tl.where(scale == 0xFF, float("nan"), mxfp)

    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(mxfp_ptr + offsets, tl.ravel(mxfp), mask=offsets < N * 32)


