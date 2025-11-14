# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import pytest
import torch

import triton
import triton.language as tl
import os
######################################## Imports ######################################## 

@triton.jit
def block_copy_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr, padding_option: tl.constexpr):
    pid = tl.program_id(0)
    # We only copy half of the data to see if the padding works
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(N // 2, ), strides=(1, ), offsets=(pid * BLOCK_SIZE, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(N, ), strides=(1, ), offsets=(pid * BLOCK_SIZE, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    if padding_option is None:
        a = tl.load(a_block_ptr, boundary_check=(0, ))
    else:
        a = tl.load(a_block_ptr, boundary_check=(0, ), padding_option=padding_option)
    tl.store(b_block_ptr, a, boundary_check=(0, ))

