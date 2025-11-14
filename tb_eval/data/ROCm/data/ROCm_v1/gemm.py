# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import sys
import argparse
import pytest
import re

# This is a Triton kernel for matrix multiplication (GEMM) with support for various data types and scaling modes.

#################### Helper utils functions ####################
# Activation function.  
@triton.jit  
def leaky_relu(x):  
    x = x + 1  
    return tl.where(x >= 0, x, 0.01 * x)  
#################### Helper utils functions ####################



@triton.autotune(  
    configs=[  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,  
                'kpack': 2, 'matrix_instr_nonkdim': 16  
            }, num_warps=4, num_stages=2),  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,  
                'kpack': 2, 'matrix_instr_nonkdim': 16  
            }, num_warps=8, num_stages=2),  
        triton.Config(  
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},  
            num_warps=8, num_stages=2),  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,  
                'kpack': 1, 'matrix_instr_nonkdim': 16  
            }, num_warps=8, num_stages=2),  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0,  
                'kpack': 1  
            }, num_warps=8, num_stages=2),  
        triton.Config(  
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},  
            num_warps=8, num_stages=2),  
        triton.Config(  
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},  
            num_warps=8, num_stages=2),  
    ],  
    key=['M', 'N', 'K'],  
    use_cuda_graph=True,  
)  
@triton.heuristics({  
    'EVEN_K':  
    lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0, 'GRID_MN':  
    lambda args: triton.cdiv(args['M'], args['BLOCK_SIZE_M']) * triton.cdiv(args['N'], args['BLOCK_SIZE_N'])  
})  
@triton.jit  
def matmul_kernel(  
    a_ptr,  
    b_ptr,  
    c_ptr,  
    M,  
    N,  
    K,  
    stride_am,  
    stride_ak,  
    stride_bk,  
    stride_bn,  
    stride_cm,  
    stride_cn,  
    a_scale_ptr,  
    b_scale_ptr,  
    stride_ascale_m,  
    stride_ascale_k,  
    stride_bscale_k,  
    stride_bscale_n,  
    # Meta-parameters  
    GROUP_K: tl.constexpr,  
    GROUP_N: tl.constexpr,  
    BLOCK_SIZE_M: tl.constexpr,  
    BLOCK_SIZE_N: tl.constexpr,  
    BLOCK_SIZE_K: tl.constexpr,  
    EVEN_K: tl.constexpr,  
    GROUP_SIZE_M: tl.constexpr,  
    APPLY_SCALE: tl.constexpr,  
    ACTIVATION: tl.constexpr,  
    GRID_MN: tl.constexpr,  
):  
    """Kernel for computing the matmul C = A x B.  
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)  
    """  
  
    NUM_XCDS: tl.constexpr = 8  
  
    tl.static_assert(((APPLY_SCALE is None) or (APPLY_SCALE == 'tensor')) or (APPLY_SCALE == 'block'),  
                     f"Scaling mode {APPLY_SCALE} is not supported!!!")  
  
    tl.assume(stride_am > 0)  
    tl.assume(stride_ak > 0)  
    tl.assume(stride_bk > 0)  
    tl.assume(stride_bn > 0)  
    tl.assume(stride_cm > 0)  
    tl.assume(stride_cn > 0)  
    tl.assume(stride_ascale_m > 0)  
    tl.assume(stride_ascale_k > 0)  
    tl.assume(stride_bscale_k > 0)  
    tl.assume(stride_bscale_n > 0)  
  
    # -----------------------------------------------------------  
    # Map program ids `pid` to the block of C it should compute.  
    # This is done in a grouped ordering to promote L2 data reuse.  
    # TODO(vgokhale): Add XCD remapping.  
    pid = tl.program_id(axis=0)  
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  
  
    ## pid remapping on xcds  
    # Number of pids per XCD in the new arrangement  
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS  
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have  
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.  
    # We calculate the number of xcds that have pids_per_xcd pids as  
    # tall_xcds  
    tall_xcds = GRID_MN % NUM_XCDS  
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds  
    # Compute current XCD and local pid within the XCD  
    xcd = pid % NUM_XCDS  
    local_pid = pid // NUM_XCDS  
    # Calculate new pid based on the new grouping  
    # Note that we need to consider the following two cases:  
    # 1. the current pid is on a tall xcd  
    # 2. the current pid is on a short xcd  
    if xcd < tall_xcds:  
        pid = xcd * pids_per_xcd + local_pid  
    else:  
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid  
  
    if GROUP_SIZE_M == 1:  
        pid_m = pid // num_pid_n  
        pid_n = pid % num_pid_n  
    else:  
        num_pid_in_group = GROUP_SIZE_M * num_pid_n  
        group_id = pid // num_pid_in_group  
        first_pid_m = group_id * GROUP_SIZE_M  
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  
        pid_m = first_pid_m + (pid % group_size_m)  
        pid_n = (pid % num_pid_in_group) // group_size_m  
  
    tl.assume(pid_m > 0)  
    tl.assume(pid_n > 0)  
  
    # Create pointers for first block of A and B input matrices  
    offs_k = tl.arange(0, BLOCK_SIZE_K)  
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M  
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)  
    if APPLY_SCALE == 'tensor':  
        a_scale = tl.load(a_scale_ptr) if a_scale_ptr else 1.0  
        b_scale = tl.load(b_scale_ptr)  
    elif APPLY_SCALE == 'block':  
        k_start = 0  
        offs_ks = k_start // GROUP_K  
        a_scale_ptrs = None if a_scale_ptr is None else (a_scale_ptr + offs_am * stride_ascale_m +  
                                                         offs_ks * stride_ascale_k)  
        offs_bsn = offs_bn // GROUP_N  
        b_scale_ptrs = b_scale_ptr + offs_bsn * stride_bscale_n + offs_ks * stride_bscale_k  
  
    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32  
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)  
  
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  
        # Load the next block of A and B, generate a mask by checking the K dimension.  
        # If it is out of bounds, set it to 0.  
        if EVEN_K:  
            a = tl.load(a_ptrs)  
            b = tl.load(b_ptrs)  
        else:  
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)  
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)  
  
        if APPLY_SCALE == 'block':  
            b_scale = tl.load(b_scale_ptrs)  
            if a_scale_ptrs is not None:  
                a_scale = tl.load(a_scale_ptrs)  
  
        # Type conversion to support mixed precision GEMMs where b is lower precision than a  
        b = b.to(a_ptr.type.element_ty)  
  
        if APPLY_SCALE == 'block':  
            if a_scale_ptrs is not None:  
                accumulator += tl.dot(a, b, input_precision="ieee") * a_scale[:, None] * b_scale[None, :]  
            else:  
                accumulator += tl.dot(a, b, input_precision="ieee") * b_scale[None, :]  
        else:  
            accumulator += tl.dot(a, b, input_precision="ieee")  
  
        # Advance the ptrs to the next K block.  
        a_ptrs += BLOCK_SIZE_K * stride_ak  
        b_ptrs += BLOCK_SIZE_K * stride_bk  
  
        if APPLY_SCALE == 'block':  
            k_cur = k * BLOCK_SIZE_K // GROUP_K  
            k_nxt = (k + 1) * BLOCK_SIZE_K // GROUP_K  
            offs_ks = k_nxt - k_cur  
            b_scale_ptrs += offs_ks * stride_bscale_k  
            if a_scale_ptrs is not None:  
                a_scale_ptrs += offs_ks * stride_ascale_k  
  
    # Apply scale to recover dynamic range reduced due to lower precision inputs.  
    if APPLY_SCALE == 'tensor':  
        accumulator = accumulator * a_scale * b_scale  
    # Apply activation function, if specified.  
    # TODO(vgokhale): Add different types of activations.  
    if ACTIVATION == "leaky_relu":  
        accumulator = leaky_relu(accumulator)  
    c = accumulator.to(c_ptr.type.element_ty)  
  
    # Write back the block of the output matrix C with masks.  
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]  
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)  
    tl.store(c_ptrs, c, mask=c_mask)  
  
  



