# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ########################################
import multiprocessing
import shutil
import tempfile
import os
import pytest

import triton
import triton.language as tl
from triton.backends.compiler import AttrsDescriptor
from triton.compiler import ASTSource
######################################## Imports ########################################

@triton.jit
def kernel_sub(a, b, o, N: tl.constexpr):
    idx = tl.arange(0, N)
    tl.store(o + idx, tl.load(a + idx) - tl.load(b + idx) * 777)


