######################################## Imports ######################################## 
import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl
######################################## Imports ######################################## 

@triton.jit
def iv_dependent_matmul(a_ptr, b_ptr, c_ptr,  #
            M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
            type: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptr = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptr = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_ptrs = a_ptr
    b_ptrs = b_ptr
    if type == "post_load_two_iters":
        a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
        b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
    elif type == "post_load_three_iters":
        a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
        b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
        a_ptrs_next_next = a_ptr + 2 * BLOCK_SIZE_K * stride_ak
        b_ptrs_next_next = b_ptr + 2 * BLOCK_SIZE_K * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if type == "pre_load":
            a_ptrs = a_ptr + k * BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptr + k * BLOCK_SIZE_K * stride_bk
        elif type == "post_pre_mixed":
            a_ptrs = a_ptr + k * BLOCK_SIZE_K * stride_ak
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        if type == "post_load":
            a_ptrs = a_ptr + (k + 1) * BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptr + (k + 1) * BLOCK_SIZE_K * stride_bk
        elif type == "post_pre_mixed":
            b_ptrs = b_ptr + (k + 1) * BLOCK_SIZE_K * stride_bk
        elif type == "post_load_two_iters":
            a_ptrs = a_ptrs_next
            b_ptrs = b_ptrs_next
            a_ptrs_next = a_ptr + (k + 2) * BLOCK_SIZE_K * stride_ak
            b_ptrs_next = b_ptr + (k + 2) * BLOCK_SIZE_K * stride_bk
        elif type == "post_load_three_iters":
            a_ptrs = a_ptrs_next
            b_ptrs = b_ptrs_next
            a_ptrs_next = a_ptrs_next_next
            b_ptrs_next = b_ptrs_next_next
            a_ptrs_next_next = a_ptr + (k + 3) * BLOCK_SIZE_K * stride_ak
            b_ptrs_next_next = b_ptr + (k + 3) * BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState


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



@pytest.mark.parametrize("type",
                         ["pre_load", "post_load", "post_pre_mixed", "post_load_two_iters", "post_load_three_iters"])
def test_iv_dependent_matmul(type, request, device='cuda'):

    
    set_seed()

    M = 256
    K = 256
    N = 256
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 32

    a = torch.rand((M, K), device=device)
    b = torch.rand((K, N), device=device)

    torch_output = torch.mm(a, b)
    triton_output = torch.empty_like(torch_output, device=torch_output.device)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    num_stages = 4 if type == "post_load_three_iters" else 3
    iv_dependent_matmul[grid](
        a, b, triton_output, M, N, K,  #
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),  #
        triton_output.stride(0), triton_output.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, type=type,  #
        num_stages=num_stages)

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save triton_output in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = triton_output.clone().detach().cpu()
    ################################################################### 

    torch.testing.assert_close(torch_output, triton_output, rtol=1e-2, atol=1e-2)

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

def test_get_result():
    print(result_gold)

######################################## HELPERS for Eval ########################################