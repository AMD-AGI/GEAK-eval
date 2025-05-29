######################################## Imports ########################################
import pytest
import torch
import triton
import triton.language as tl
import numpy as np
######################################## Imports ########################################

@triton.jit
def flip_kernel(X, Z, N: tl.constexpr, M: tl.constexpr):
    offx = tl.arange(0, M)
    offy = tl.arange(0, N) * M
    off2d = offx[None, :] + offy[:, None]
    x = tl.load(X + off2d)
    x = tl.flip(x)
    tl.store(Z + off2d, x)


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




# Common helper code (similar to test_triton_sort.py)
def np_dtype_from_str(dtype_str):
    if dtype_str == "bfloat16": return np.float32
    return getattr(np, dtype_str)

def gen_numpy_array_for_torch_conversion(shape, dtype_str, low=0, high=100):
    np_dtype = np_dtype_from_str(dtype_str)
    actual_low = int(low)
    actual_high = int(high)

    if dtype_str == 'bfloat16':
        return np.random.uniform(actual_low, actual_high, shape).astype(np.float32)
    if 'float' in dtype_str:
        return np.random.uniform(actual_low, actual_high, shape).astype(np_dtype)
    elif 'int' in dtype_str:
        if actual_high <= actual_low: actual_high = actual_low + 1
        if np.issubdtype(np_dtype, np.integer):
            iinfo = np.iinfo(np_dtype)
            actual_low = max(iinfo.min, actual_low)
            effective_high = min(iinfo.max, actual_high)
            if effective_high <= actual_low: effective_high = actual_low +1
            if effective_high > iinfo.max : effective_high = iinfo.max
            return np.random.randint(actual_low, effective_high + 1 if effective_high < iinfo.max else effective_high, shape, dtype=np_dtype)
    raise ValueError(f"Unsupported dtype_str for gen_numpy_array_for_torch_conversion: {dtype_str}")

def torch_dtype_from_str(dtype_str):
    if dtype_str == "bfloat16": return torch.bfloat16
    if dtype_str == "float16": return torch.float16
    if dtype_str == "float32": return torch.float32
    if dtype_str == "int32": return torch.int32
    raise ValueError(f"Unsupported dtype_str for torch: {dtype_str}")


@pytest.mark.interpreter
@pytest.mark.parametrize("M_cols, N_rows", [[512, 1], [64, 8], [16, 256], [8, 512]])
@pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32', 'bfloat16'])
def test_flip(N_rows, M_cols, dtype_str, request, device='cuda'):
    set_seed()

    x_np = gen_numpy_array_for_torch_conversion((N_rows, M_cols), dtype_str=dtype_str)
    
    torch_dtype = torch_dtype_from_str(dtype_str)
    x = torch.from_numpy(x_np).to(dtype=torch_dtype, device=device)

    y_ref = torch.flip(x, dims=(1,)) # Flip along the columns (dimension 1)
    z_triton = torch.empty_like(x)

    grid = (N_rows,) # Each program flips one row
    flip_kernel[grid](x, z_triton, N_rows, M_cols) # num_warps removed

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])

    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = z_triton.clone().detach().cpu()
    ################################################################### 

    assert torch.allclose(y_ref.float(), z_triton.float(), atol=1e-2, rtol=1e-2), \
        f"Flip mismatch for dtype {dtype_str}, shape ({N_rows}, {M_cols}).\nReference: {y_ref}\nTriton: {z_triton}"



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