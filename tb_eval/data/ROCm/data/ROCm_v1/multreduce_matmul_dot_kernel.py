# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
# Imports:  
# --------  
  
import argparse  
import itertools  
import os  
import sys  
from typing import Any, Callable, Optional  
  
import pytest  
import torch  
from torch import Tensor  
  
import triton  
import triton.language as tl  
 

######################## HELPER UTILS #####################
# Autotune configurations for Triton GEMM implemented with `tl.dot`.  
def get_triton_dot_autotune_configs() -> list[triton.Config]:  
    block_size_n_range: list[int] = [16, 32]  
    block_size_k_range: list[int] = [128, 256, 512]  
    kpack_range: list[int] = [1, 2]  
    num_warps_range: list[int] = [1, 2]  
    return [  
        triton.Config(  
            {  
                "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k, "waves_per_eu": 0,  
                "matrix_instr_nonkdim": 16, "kpack": kpack  
            }, num_warps=num_warps, num_stages=2) for block_size_n, block_size_k, kpack, num_warps in itertools.product(  
                block_size_n_range, block_size_k_range, kpack_range, num_warps_range)  
    ]  
  
  
def get_triton_autotune_key() -> list[str]:  
    return ["M", "N", "K"]  
  
  
def get_triton_heuristics() -> dict[str, Callable[[dict[str, Any]], Any]]:  
    return {"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0}  

###############################################################
# Triton GEMM:  
# ------------  
# Core Triton GEMM kernel.  
# Triton GEMM kernel implemented with `tl.dot`.  
@triton.autotune(configs=get_triton_dot_autotune_configs(), key=get_triton_autotune_key())  
@triton.heuristics(get_triton_heuristics())  
@triton.jit  
def triton_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #  
                         M: int, N: int, K: int,  #  
                         stride_am: int, stride_ak: int,  #  
                         stride_bk: int, stride_bn: int,  #  
                         stride_cm: int, stride_cn: int,  #  
                         stride_bias: int,  #  
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #  
                         USE_BIAS: tl.constexpr, USE_DOT: tl.constexpr, EVEN_K: tl.constexpr  #  
                         ):  
    # Compute program ID:  
    pid = tl.program_id(axis=0)  
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  
    pid_m = pid // num_pid_n  
    pid_n = pid % num_pid_n  
  
    # Compute A and B base pointers:  
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  
    offs_k = tl.arange(0, BLOCK_SIZE_K)  
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak  
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn  
  
    # Load BIAS:  
    if USE_BIAS:  
        bias_ptrs = bias_ptr + offs_am * stride_bias  
        bias = tl.load(bias_ptrs, mask=offs_am < M, other=0)  
  
    # Initialize accumulator:  
    acc_dtype = tl.float32 if a_ptr.type.element_ty != tl.int8 else tl.int32  
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)  
  
    # GEMM loop:  
  
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  
        if EVEN_K:  
            # Unmasked load of A and B:  
            a = tl.load(a_ptrs)  
            b = tl.load(b_ptrs)  
        else:  
            # Masked load of A and B:  
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)  
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)  
        # Compute dot product:  
        if USE_DOT:  
            accumulator += tl.dot(a, b)  
        else:  
            a = tl.reshape(a, (BLOCK_SIZE_M, BLOCK_SIZE_K, 1)).to(acc_dtype)  
            b = tl.reshape(b, (1, BLOCK_SIZE_K, BLOCK_SIZE_N)).to(acc_dtype)  
            accumulator += tl.sum(a * b, axis=1)  
        # Advance A and B pointers:  
        a_ptrs += BLOCK_SIZE_K * stride_ak  
        b_ptrs += BLOCK_SIZE_K * stride_bk  
  
    # Convert accumulator back to C's type:  
    c = accumulator.to(c_ptr.type.element_ty)  
  
    # Add BIAS:  
    if USE_BIAS:  
        c += bias[:, None]  
  
    # Compute C pointers and store C:  
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn  
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)  
    tl.store(c_ptrs, c, mask=c_mask)  
  
  
# # Triton GEMM kernel implemented with `tl.dot`.  
# @triton.autotune(configs=get_triton_dot_autotune_configs(), key=get_triton_autotune_key())  
# @triton.heuristics(get_triton_heuristics())  
# @triton.jit  
# def triton_dot_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #  
#                              M: int, N: int, K: int,  #  
#                              stride_am: int, stride_ak: int,  #  
#                              stride_bk: int, stride_bn: int,  #  
#                              stride_cm: int, stride_cn: int,  #  
#                              stride_bias: int,  #  
#                              BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #  
#                              USE_BIAS: tl.constexpr, EVEN_K: tl.constexpr  #  
#                              ):  
#     triton_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #  
#                          M, N, K,  #  
#                          stride_am, stride_ak,  #  
#                          stride_bk, stride_bn,  #  
#                          stride_cm, stride_cn,  #  
#                          stride_bias,  #  
#                          BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #  
#                          USE_BIAS=USE_BIAS, USE_DOT=True, EVEN_K=EVEN_K)  
 

  
  
