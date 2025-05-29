######################################## Imports ######################################## 
import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl

######################################## Imports ######################################## 

@triton.jit
def batched_vecmat(
        # inputs
        A,  # shape: [dim_m, dim_k]
        B,  # shape: [dim_m, dim_n, dim_k]
        # dimensions
    dim_m, dim_n, dim_k,
        # outputs
        output,
        # block information
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)
    # Output tile
    output_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_n \
        + (n_index * block_n + tl.arange(0, block_n))[None, :]

    vecmat = tl.zeros([block_m, block_n], dtype=A.dtype.element_ty)
    k_blocks = dim_k // block_k
    for k_index in range(k_blocks):
        # Load A tile
        a_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, :]
        a = tl.load(A + a_tile)

        # Load B tile, transposed to [n, m, k] in order to broadcast A on a
        # leading dimension.
        b_tile = (m_index * block_m + tl.arange(0, block_m))[None, :, None] * dim_n * dim_k \
            + (n_index * block_n + tl.arange(0, block_n))[:, None, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, None, :]
        b = tl.load(B + b_tile)

        expanded_a, _ = tl.broadcast(a, b)
        vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))

    tl.store(output + output_tile, vecmat)


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

  
def test_vecmat(request, device='cuda'):  
    """  
    Test the batched vector-matrix multiplication kernel.  
  
    Args:  
        device: The device (e.g., 'cuda' or 'cpu') on which the test is executed.  
        request: Pytest request object used to retrieve the test case name.  
    """  
    set_seed()  
  
    M, N, K = 128, 128, 128  
    block_m, block_n, block_k = 16, 32, 64  
  
    rs = RandomState(17)  
    A_vec = rs.randint(0, 4, (M, K)).astype('float32')  
    B_vec = rs.randint(0, 4, (M, N, K)).astype('float32')  
    A = A_vec  
    B = B_vec  
  
    A_tri = torch.tensor(A, device=device)  
    B_tri = torch.tensor(B, device=device)  
    C_tri = torch.zeros((M, N), dtype=torch.float32, device=device)  
  
    grid = (M // block_m, N // block_n)  
  
    # This is where the actual kernel would run and populate C_tri  
    # If using the MockKernel above, C_tri will remain zeros unless populated for testing.  
    # For the purpose of demonstrating saving, we'll assume C_tri is populated by the kernel.  
    # To make the assert_allclose pass without a real kernel, we can compute C_tri using torch:  
    if isinstance(batched_vecmat, MockKernel): # If using the mock  
        A_expanded_torch = A_tri[:, None, :] # (M, 1, K)  
        # B_tri is (M, N, K)  
        # Element-wise product and sum over K  
        # This is what the kernel is supposed to compute  
        C_tri_computed_by_torch = torch.sum(A_expanded_torch * B_tri, dim=2) # (M, N)  
        C_tri.copy_(C_tri_computed_by_torch) # Populate C_tri as the kernel would  
    else: # If using the real kernel  
        batched_vecmat[grid](  
            A_tri, B_tri, M, N, K, C_tri,  #  
            block_m=block_m, block_n=block_n, block_k=block_k,  #  
            num_warps=4, num_stages=1)  
  
  
    A_expanded_np = A[:, np.newaxis, :] # (M, 1, K)  
    A_broadcasted_np = np.broadcast_to(A_expanded_np, (M, N, K)) # (M, N, K)  
    AB_np = A_broadcasted_np * B # B is (M, N, K)  
    C_ref = np.sum(AB_np, axis=2) # (M, N)  
  
    ################### save C_ref and C_tri as torch tensors in result_gold ###################  
    test_case_name = request.node.name  
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")  
  
    # Convert C_ref (numpy array) to a torch tensor on CPU  
    # C_ref is float32 because A_vec and B_vec are float32.  
    C_ref_torch = torch.tensor(C_ref, dtype=torch.float32, device='cpu')  
  
    # Store C_ref_torch and C_tri (on CPU) in result_gold  
    # C_tri is already torch.float32 as defined.  
    result_gold[sanitized_key_name + "_ref"] = C_ref_torch  
    result_gold[sanitized_key_name + "_tri"] = C_tri.cpu() # Ensure C_tri is on CPU for saving  
    #########################################################################################  
  
    np.testing.assert_allclose(C_ref, C_tri.cpu().numpy(), rtol=0.01, atol=1e-3)  
  


def test_vecmat(request, device='cuda'):
    """
    Test the batched vector-matrix multiplication kernel.

    Args:
        device: The device (e.g., 'cuda' or 'cpu') on which the test is executed.
        request: Pytest request object used to retrieve the test case name.
    """
    set_seed()

    M, N, K = 128, 128, 128
    block_m, block_n, block_k = 16, 32, 64

    rs = RandomState(17)
    A_vec = rs.randint(0, 4, (M, K)).astype('float32')
    B_vec = rs.randint(0, 4, (M, N, K)).astype('float32')
    A = A_vec
    B = B_vec

    A_tri = torch.tensor(A, device=device)
    B_tri = torch.tensor(B, device=device)
    C_tri = torch.zeros((M, N), dtype=torch.float32, device=device)

    grid = (M // block_m, N // block_n)

    batched_vecmat[grid](
        A_tri, B_tri, M, N, K, C_tri,  #
        block_m=block_m, block_n=block_n, block_k=block_k,  #
        num_warps=4, num_stages=1)

    A_expanded = A[:, np.newaxis, :]
    A_broadcasted = np.broadcast_to(A_expanded, (M, N, K))
    AB = A_broadcasted * B
    C_ref = np.sum(AB, axis=2)

    ################### save tri_out in result_gold ###################
    # Convert C_ref (numpy array) to a torch tensor on CPU  
    # C_ref is float32 because A_vec and B_vec are float32.  

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ## triton_result is assumed to be torch tensor not numpy ndarray
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = C_tri.cpu()
    ################################################################### 
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    np.testing.assert_allclose(C_ref, C_tri.cpu().numpy(), rtol=0.01, atol=1e-3)

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

def test_get_results():
    print(result_gold)
######################################## HELPERS for Eval ########################################