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
def kernel_dot(Z):
    offs = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    z = tl.load(Z + offs)
    z = tl.dot(z, z)
    tl.store(Z + offs, z)


