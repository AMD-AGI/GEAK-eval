######################################## Imports ######################################## 
import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl

######################################## Imports ######################################## 


@triton.jit
def chained_matmul_kernel(A,  # shape: (m, k)
                            B,  # shape: (n, k)
                            C,  # shape: (n, k)
                            out,  # shape: (m, k)
                            m, n, k: tl.constexpr,  #
                            block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):

    tl.static_assert(block_k == k, f"expected block_k == k but got {block_k} != {k}")

    block_ix = tl.program_id(0)
    a_tile = (block_ix * block_m + tl.arange(0, block_m))[:, None] * block_k \
        + tl.arange(0, block_k)[None, :]

    a = tl.load(A + a_tile, mask=a_tile < m * k, other=0.0)

    acc = tl.zeros([block_m, block_k], dtype=tl.float32)

    for loop_block_start in range(0, n, block_n):
        bc_tile = (loop_block_start + tl.arange(0, block_n))[:, None] * block_k \
            + tl.arange(0, block_k)[None, :]
        b = tl.load(B + bc_tile, mask=bc_tile < n * k, other=0.0)

        intermediate = tl.dot(a, tl.trans(b))
        intermediate_mask = ((loop_block_start + tl.arange(0, block_n)) < n)[None, :] \
            * (tl.arange(0, block_m) < m)[:, None]

        intermediate = tl.where(intermediate_mask, intermediate, 0.0)

        c = tl.load(C + bc_tile, mask=bc_tile < n * k)

        acc += tl.dot(intermediate.to(A.dtype.element_ty), c)

    tl.store(out + a_tile, acc.to(A.dtype.element_ty), mask=a_tile < m * k)


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


def chained_matmul_reference(a, b, c):
    intermediate = torch.einsum('MK,NK->MN', a, b)
    return torch.einsum('MN,NK->MK', intermediate, c)


def test_chained_matmul(request, device='cuda'):
    # Regression test for issue #1601
    set_seed()


    m, n, k = 32, 64, 128
    block_m, block_n, block_k = 16, 32, k

    grid = (triton.cdiv(m, block_m), )
    a = torch.randint(low=0, high=2, size=(m, k), dtype=torch.float16, device=device)
    b = torch.randint(low=0, high=2, size=(n, k), dtype=torch.float16, device=device)
    c = torch.randint_like(b, low=0, high=2)
    triton_result = torch.zeros_like(a)

    torch_result = chained_matmul_reference(a, b, c)
    chained_matmul_kernel[grid](
        a, b, c, triton_result, m, n, k,  #
        block_m=block_m, block_n=block_n, block_k=block_k)

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = triton_result.clone().detach().cpu()
    ################################################################### 
    assert (torch_result == triton_result).all()


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