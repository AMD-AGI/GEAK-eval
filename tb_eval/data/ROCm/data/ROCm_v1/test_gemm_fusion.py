# Usage Instruction: python3 -m pytest test_gemm_fusion.py

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


######################################## Imports#######################################
import pytest
import torch

import triton
import triton.language as tl

######################################## Imports#######################################



@triton.jit
def gemm_fusion_kernel(A, B, C, E,  #
                       M, N, K,  #
                       stride_am, stride_ak, stride_bn, stride_bk, stride_cn, stride_ck, stride_em, stride_ek,  #
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)

    a_tile_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak), offsets=(pid * BLOCK_M, 0),
                                   block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_tile_ptr = tl.make_block_ptr(base=B, shape=(N, K), strides=(stride_bn, stride_bk), offsets=(0, 0),
                                   block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))
    c_tile_ptr = tl.make_block_ptr(base=C, shape=(N, K), strides=(stride_cn, stride_ck), offsets=(0, 0),
                                   block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))
    e_tile_ptr = tl.make_block_ptr(base=E, shape=(M, K), strides=(stride_em, stride_ek), offsets=(pid * BLOCK_M, 0),
                                   block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))

    acc_e = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    a = tl.load(a_tile_ptr)
    for i in range(0, N, BLOCK_N):
        b = tl.load(b_tile_ptr)
        o_ab = tl.dot(a, tl.trans(b))
        c = tl.load(c_tile_ptr)
        o_ab = o_ab.to(tl.float16)
        acc_e += tl.dot(o_ab, c)
        b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_N, 0])
        c_tile_ptr = tl.advance(c_tile_ptr, [BLOCK_N, 0])

    acc_e = acc_e.to(tl.float16)
    tl.store(e_tile_ptr, acc_e)

##################################################################################################################################################  
import numpy as np
import random
import torch 
import os


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

######################################## HELPERS for Eval ########################################


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="not passed on ampere")
def test_gemm_fusion(request):
    set_seed()


    M, N, K = 4096, 4096, 64
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    A = torch.empty((M, K), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    B = torch.empty((N, K), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    C = torch.empty((N, K), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    E = torch.empty((M, K), dtype=torch.float16, device='cuda')
    ref_out = torch.matmul(torch.matmul(A, B.T), C)
    num_warps = 4
    grid = (triton.cdiv(M, BLOCK_M), 1)
    gemm_fusion_kernel[grid](
        A, B, C, E, M, N, K,  #
        A.stride(0), A.stride(1),  #
        B.stride(0), B.stride(1),  #
        C.stride(0), C.stride(1),  #
        E.stride(0), E.stride(1),  #
        BLOCK_M, BLOCK_N, BLOCK_K,  #
        num_warps=num_warps)

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_") 
    result_gold[sanitized_key_name] = E.clone().detach().cpu()
    ###################################################################

    torch.testing.assert_close(ref_out, E, atol=1e-2, rtol=1e-2)


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