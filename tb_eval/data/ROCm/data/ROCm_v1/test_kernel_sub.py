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


##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState
import multiprocessing
import shutil
import tempfile
import os
import pytest

import triton
import triton.language as tl
from triton.backends.compiler import AttrsDescriptor
from triton.compiler import ASTSource

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



# --- Fixture Definition ---
@pytest.fixture
def fresh_triton_cache(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    cache_dir = os.path.join(temp_dir, ".triton")
    os.makedirs(cache_dir, exist_ok=True)
    monkeypatch.setenv("TRITON_CACHE_DIR", cache_dir)
    yield cache_dir
    shutil.rmtree(temp_dir)
# --- End Fixture Definition ---

# Check if a target is available. Skip tests if not.
try:
    target = triton.runtime.driver.active.get_current_target()
    TARGET_AVAILABLE = True
except Exception:
    TARGET_AVAILABLE = False
    target = None

# Decorator to skip tests if target is not available
skip_if_no_target = pytest.mark.skipif(not TARGET_AVAILABLE, reason="Triton target not available (e.g., no GPU or CUDA/ROCm setup)")

@skip_if_no_target
def compile_kernel_sub_for_test(attrs): # Renamed to be specific
    # kernel_sub is defined globally above
    src = ASTSource(
        fn=kernel_sub,
        constants={'N': 32},
        signature={'a': "*fp32", 'b': "*fp32", 'o': "*fp32"},
        attrs=attrs,
    )
    triton.compile(src=src, target=target)

@skip_if_no_target
def test_compile_kernel_sub_in_subproc(fresh_triton_cache, request) -> None: # Test name updated for clarity

    set_seed()
    
    config = AttrsDescriptor.from_hints({i: 16 for i in range(4)})
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        print("Warning: Could not force 'fork' start method. Using default.")
        if multiprocessing.get_start_method(allow_none=True) != 'fork': # allow_none for safety
            pytest.skip("Test requires 'fork' multiprocessing start method.")

    proc = multiprocessing.Process(target=compile_kernel_sub_for_test, args=(config, ))
    proc.start()
    proc.join(timeout=60)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        pytest.fail("Process timed out")

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])

    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = torch.tensor([[0.0]]).clone().detach().cpu()
    ################################################################### 

    assert proc.exitcode == 0

    result_gold[sanitized_key_name] = torch.tensor([[1.0]]).clone().detach().cpu()

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