# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import argparse  
import sys  
import pytest  
  
import torch  
import triton  
import triton.language as tl  
import os  
import json  
import math  
from itertools import product  
  
######################################## HELPERS utils ######################################## 
def is_cuda():  
    return triton.runtime.driver.active.get_current_target().backend == "cuda"  
  
  
def is_hip():  
    return triton.runtime.driver.active.get_current_target().backend == "hip"  
  
  
def get_cuda_autotune_config():  
    return [  
        triton.Config({}, num_warps=4, num_stages=1),  
        triton.Config({}, num_warps=8, num_stages=1),  
        triton.Config({}, num_warps=16, num_stages=1),  
    ]  
  
  
def get_hip_autotune_config():  
    return [  
        triton.Config({'waves_per_eu': we}, num_warps=wa, num_stages=1) for we, wa in product([1, 2, 4], [4, 8, 16])  
    ]  
  
  
def get_autotune_config():  
    if is_cuda():  
        return get_cuda_autotune_config()  
    else:  
        return get_hip_autotune_config()  
  
######################################## HELPERS utils ######################################## 


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)  
@triton.jit  
def layernorm_kernel(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, x_row_stride, y_row_stride, n_rows, n_cols, eps,  
                     BLOCK_SIZE: tl.constexpr):  
  
    #program id  
    row = tl.program_id(0)  
    x_ptr_start = x_ptr + (row * x_row_stride)  
    y_ptr_start = y_ptr + (row * y_row_stride)  
  
    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1  
  
    #calculate mean  
    mean = 0  
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  
    loop_num_l = loop_num  
    for b in range(0, loop_num_l):  
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  #Unmasked loads  
        _mean += x_block  
  
    #For last iteration, do masked load  
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
    x_block = tl.load(x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.).to(tl.float32)  
    _mean += x_block  
    mean = tl.sum(_mean, axis=0) / n_cols  
  
    #variance  
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  
    loop_num_l = loop_num  
    for b in range(0, loop_num_l):  
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  #Unmasked loads  
        x_block = x_block - mean  
        _var += x_block * x_block  
  
    #For last iteration, do masked load  
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
    x_block = tl.load(x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.).to(tl.float32)  
    x_block = tl.where(col_offsets < n_cols, x_block - mean, 0.)  
    _var += x_block * x_block  
  
    var = tl.sum(_var, axis=0) / n_cols  
    rstd = tl.rsqrt(var + eps)  
  
    # Write mean / rstd  
    tl.store(mean_ptr + row, mean)  
    tl.store(rstd_ptr + row, rstd)  
  
    #Normalize and store  
    loop_num_l = loop_num  
    for b in range(0, loop_num_l):  
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
        w_block = tl.load(w_ptr + col_offsets)  
        b_block = tl.load(b_ptr + col_offsets)  
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  
        y_block = (x_block - mean) * rstd  
        y_block = y_block * w_block + b_block  
        tl.store(y_ptr_start + col_offsets, y_block)  
  
    #For last iteration, do masked load and store  
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
    mask = col_offsets < n_cols  
    w_block = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)  
    b_block = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)  
    x_block = tl.load(x_ptr_start + col_offsets, mask=mask, other=0.0).to(tl.float32)  
    y_block = (x_block - mean) * rstd  
    y_block = y_block * w_block + b_block  
    tl.store(y_ptr_start + col_offsets, y_block, mask=mask)  
  
  
