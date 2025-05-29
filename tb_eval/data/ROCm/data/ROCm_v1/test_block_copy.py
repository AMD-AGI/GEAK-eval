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

##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState

result_gold = {}

######################################## HELPERS for Eval ######################################## 
def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'

def check_type_supported(dtype, device='cuda'):
    '''
    skip test if dtype is not supported on the current device
    '''
    if device in ['cuda']:
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8 and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
            pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")
        if cc[0] < 9 and dtype in {tl.float8e4nv, "float8e4nv", "float8_e4m3fn"}:
            pytest.skip("float8e4nv is only supported on NVGPU with cc >= 90")
    if is_interpreter():
        if dtype in [tl.bfloat16, "bfloat16", torch.bfloat16]:
            pytest.skip("bfloat16 is not supported in the interpreter")


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



@pytest.mark.interpreter
@pytest.mark.parametrize("dtypes_str, n, padding_option", [  #
    (dtypes_str, n, padding)
    for dtypes_str in (("bool", "bool"), ("int16", "int16"), ("int32", "int32"), ("float16", "float16"),
                       ("float32", "float32"), ("bfloat16", "bfloat16"))
    for n in (64, 128, 256, 512, 1024)
    for padding in (None, "zero", "nan")  #
])
def test_block_copy(dtypes_str, n, padding_option, request, device='cuda'):
    src_dtype_str = dtypes_str[0]
    dst_dtype_str = dtypes_str[1]
    src_dtype = getattr(torch, src_dtype_str)
    dst_dtype = getattr(torch, dst_dtype_str)
    check_type_supported(src_dtype, device)
    check_type_supported(dst_dtype, device)
    if src_dtype_str in ("bool", "int16", "int32"):
        if padding_option == "nan":
            pytest.skip("Padding with NaN is not supported for integer types")
        a = torch.randint(0, 2, (n, ), device=device, dtype=src_dtype)
    else:
        a = torch.randn((n, ), device=device, dtype=src_dtype)
    b = torch.zeros((n, ), device=device, dtype=dst_dtype)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
    block_copy_kernel[grid](a_ptr=a, b_ptr=b, N=n, BLOCK_SIZE=64, padding_option=padding_option)
    a.to(dst_dtype)
    
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    assert torch.all(a[0:n // 2] == b[0:n // 2])
    if padding_option == "zero":
        assert torch.all(b[n // 2:n] == 0)
    elif padding_option == "nan":
        assert torch.all(torch.isnan(b[n // 2:n]))

    
    ################### save True in result_gold (indicates it passed block copy tests, implies the gen kernel worked) ###################
    c = torch.tensor([[1.0]])
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = c.clone().detach().cpu()
    ################################################################### 


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