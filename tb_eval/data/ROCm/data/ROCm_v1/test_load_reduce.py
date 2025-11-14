# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl
######################################## Imports ######################################## 

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
}


@triton.jit
def load_reduce_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    x_ptr = tl.make_block_ptr(base=x_ptr, shape=(BLOCK_M, BLOCK_N), strides=(stride_xm, stride_xn), offsets=(0, 0),
                              block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    x = tl.load(x_ptr)
    y = tl.max(x, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_M), y)


