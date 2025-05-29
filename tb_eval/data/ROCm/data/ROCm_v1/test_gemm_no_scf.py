# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
######################################## Imports ######################################## 
import itertools
import os
import re

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

######################################## Imports ######################################## 



@triton.jit
def matmul_no_scf_kernel(a_ptr, b_ptr, c_ptr,  #
                         M, N, K,  #
                         stride_am, stride_ak,  #
                         stride_bk, stride_bn,  #
                         stride_cm, stride_cn,  #
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                         FLOAT16_OUTPUT: tl.constexpr, USE_TMA_EPILOGUE: tl.constexpr  #
                         ):
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    c = tl.dot(a, b)

    if FLOAT16_OUTPUT:
        c = c.to(tl.float16)

    if USE_TMA_EPILOGUE:
        c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
        tl.store(c_block_ptr, c)
    else:
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c)

##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState
import itertools
import re

from torch.testing import assert_close

import triton
import triton.language as tl


result_gold = {}

######################################## HELPERS for Eval ######################################## 
def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across multiple libraries and configure PyTorch for deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


######################################## HELPERS for Eval ######################################## 




@pytest.mark.parametrize(
    'M,N,K,NUM_CTAS,NUM_WARPS,TRANS_A,TRANS_B,OUTPUT_TYPE,USE_TMA_EPILOGUE',
    itertools.chain(*[[
        # numCTAs = 1, no TMA multicast:
        [64, 16, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [64, 32, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [64, 64, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [64, 64, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [64, 64, 32, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [64, 64, 64, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [128, 128, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [128, 128, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        # static mask, cluster 4x1
        [256, 64, 16, 4, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [256, 64, 16, 4, 4, False, True, "float32", USE_TMA_EPILOGUE],
        # dynamic mask, cluster 2x2
        [128, 128, 16, 4, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [128, 128, 16, 4, 4, False, True, "float32", USE_TMA_EPILOGUE],
        # small M, N
        [16, 16, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [16, 32, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [32, 16, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [32, 32, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
    ] for USE_TMA_EPILOGUE in [True, False]]))
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_gemm_no_scf(M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_TYPE, USE_TMA_EPILOGUE, request):
    set_seed()
    
    if is_hip() and NUM_CTAS > 1:
        pytest.skip("NUM_CTAS > 1 is not supported in HIP backend")

    if (TRANS_A):
        a = torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    if (TRANS_B):
        b = torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    if OUTPUT_TYPE == "float16":
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    else:
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    matmul_no_scf_kernel[(1, 1)](
        a_ptr=a, b_ptr=b, c_ptr=c,  #
        M=M, N=N, K=K,  #
        stride_am=a.stride(0), stride_ak=a.stride(1),  #
        stride_bk=b.stride(0), stride_bn=b.stride(1),  #
        stride_cm=c.stride(0), stride_cn=c.stride(1),  #
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,  #
        num_warps=NUM_WARPS,  #
        num_ctas=NUM_CTAS,  #
        FLOAT16_OUTPUT=(OUTPUT_TYPE == "float16"),  #
        USE_TMA_EPILOGUE=USE_TMA_EPILOGUE)
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    golden = torch.matmul(a_f32, b_f32)
    torch.set_printoptions(profile="full")

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = c.clone().detach().cpu()
    ###################################################################


    assert_close(c, golden, rtol=1e-2, atol=1e-3, check_dtype=False)



######################################## HELPERS for Eval ########################################     
# --- Pytest hook to save the dictionary at the end of the session ---  
def test_save_results():  
    """  
    Called after whole test run finished, right before returning the exit status to the system.  
    """
    print('Inside session finish...')
    if "_CALL_SUCCESS_" not in result_gold:
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[0.0]])
    OUTPUT_FILENAME = __file__.replace('.','_') + '.pt'
    print(f"\nSaving all y_triton results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} y_triton tensors to {OUTPUT_FILENAME}.")  

######################################## HELPERS for Eval ########################################