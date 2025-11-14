# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl
######################################## Imports ######################################## 

@triton.jit
def reverse_range(in_ptr, out_ptr):
    x0 = tl.arange(0, 512)
    tmp0 = tl.load(in_ptr + (512 - x0))
    tl.store(out_ptr + x0, tmp0)

