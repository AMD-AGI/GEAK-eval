######################################## Imports ######################################## 
import numpy as np
import pytest
import torch

import triton
import triton.language as tl
######################################## Imports ######################################## 

#####################################
# Triton Kernels for randint
#####################################

BLOCK: tl.constexpr = 1024

@triton.jit
def randint_kernel_runtime_seed(X, N, seed_val): # Kernel for runtime seed
    pid = tl.program_id(0).to(X.dtype.element_ty) # pid uses X's dtype for consistency if X is int
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    rand = tl.randint(seed_val, offset)
    tl.store(X + offset, rand, mask=offset < N)

@triton.jit
def randint_kernel_const_seed(X, N, seed_val: tl.constexpr): # Kernel for const seed
    pid = tl.program_id(0).to(X.dtype.element_ty) # pid uses X's dtype for consistency
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    rand = tl.randint(seed_val, offset)
    tl.store(X + offset, rand, mask=offset < N)

##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState
import triton
import triton.language as tl

result_gold = {}
BLOCK: tl.constexpr = 1024
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

#####################################
# Reference Philox Implementation
#####################################


class PhiloxConfig:

    def __init__(self, PHILOX_ROUND_A, PHILOX_ROUND_B, PHILOX_KEY_A, PHILOX_KEY_B, DTYPE):
        self.PHILOX_ROUND_A = np.array(PHILOX_ROUND_A, dtype=DTYPE)
        self.PHILOX_ROUND_B = np.array(PHILOX_ROUND_B, dtype=DTYPE)
        self.PHILOX_KEY_A = np.array(PHILOX_KEY_A, dtype=DTYPE)
        self.PHILOX_KEY_B = np.array(PHILOX_KEY_B, dtype=DTYPE)
        self.DTYPE = DTYPE


# This is better for GPU
PHILOX_32 = PhiloxConfig(
    PHILOX_KEY_A=0x9E3779B9,
    PHILOX_KEY_B=0xBB67AE85,
    PHILOX_ROUND_A=0xD2511F53,
    PHILOX_ROUND_B=0xCD9E8D57,
    DTYPE=np.uint32,
)

# This is what numpy implements
PHILOX_64 = PhiloxConfig(
    PHILOX_KEY_A=0x9E3779B97F4A7C15,
    PHILOX_KEY_B=0xBB67AE8584CAA73B,
    PHILOX_ROUND_A=0xD2E7470EE14C6C93,
    PHILOX_ROUND_B=0xCA5A826395121157,
    DTYPE=np.uint64,
)


class CustomPhilox4x:

    def __init__(self, seed, config):
        self._config = config
        seed = self._into_pieces(seed)
        self._key = np.array(seed[:2], dtype=self._dtype)
        self._counter = np.array((0, 0) + seed[2:], dtype=self._dtype)

    @property
    def _dtype(self):
        return self._config.DTYPE

    def _into_pieces(self, n, pad=4):
        res = []
        bits = np.dtype(self._dtype).itemsize * 8
        while len(res) < pad:
            res.append(np.array((n & ((1 << bits) - 1)), dtype=self._dtype))
            n >>= bits
        assert n == 0
        return tuple(res)

    def _multiply_low_high(self, a, b):
        low = a * b
        high = int(a) * int(b)
        high = np.array(high >> (np.dtype(self._dtype).itemsize * 8), dtype=self._dtype)
        return low, high

    def _single_round(self, counter, key):
        lo0, hi0 = self._multiply_low_high(self._config.PHILOX_ROUND_A, counter[0])
        lo1, hi1 = self._multiply_low_high(self._config.PHILOX_ROUND_B, counter[2])
        ret0 = hi1 ^ counter[1] ^ key[0]
        ret1 = lo1
        ret2 = hi0 ^ counter[3] ^ key[1]
        ret3 = lo0
        return np.array([ret0, ret1, ret2, ret3], dtype=self._dtype)

    def _raise_key(self, key):
        pk = [self._config.PHILOX_KEY_A, self._config.PHILOX_KEY_B]
        return key + np.array(pk, dtype=self._dtype)

    def random_raw(self):
        counter = self._counter
        key = self._key
        for _ in range(10):
            counter = self._single_round(counter, key)
            key = self._raise_key(key)
        self.advance(1)
        return counter

    def advance(self, n_steps):
        self._counter[0] += n_steps
        assert self._counter[0] < 2**32, "FIXME: doesn't work for large offsets"


class CustomPhilox(CustomPhilox4x):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = []

    def random_raw(self):
        if len(self.buffer) == 0:
            self.buffer = list(super().random_raw())[::-1]
        return int(self.buffer.pop())


#####################################
# Unit Test for randint
#####################################

@pytest.mark.interpreter
@pytest.mark.parametrize('size, seed, dtype, const_seed', [(size, seed, dtype, const_seed)
                                                           for size_str in ['10', '4,53', '400'] # Renamed to size_str
                                                           for size in [[int(s) for s in size_str.split(',')]] # Process size here
                                                           for seed in [0, 42, 124, 54, 0xffffffff, 0x0000000fcafeb0ba]
                                                           for dtype in ['int32', 'int64']
                                                           for const_seed in [True, False]])
def test_randint(size, seed, dtype, const_seed, request, device='cuda'):
    # size = list(map(int, size.split(','))) # Moved to parametrize
    set_seed()

    torch_dtype = getattr(torch, dtype)
    numpy_dtype = getattr(np, f"u{dtype}") # Philox generates unsigned integers
    config = {'int32': PHILOX_32, 'int64': PHILOX_64}[dtype]

    # triton result
    x = torch.empty(size, dtype=torch_dtype, device=device)
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK), )
    if N > 0: # Ensure grid is not (0,) if N is 0
        if const_seed:
            randint_kernel_const_seed[grid](x, N, seed_val=seed)
        else:
            randint_kernel_runtime_seed[grid](x, N, seed_val=seed)
    
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = x.clone().detach().cpu()
    ################################################################### 

    out_tri = x.cpu().numpy().astype(numpy_dtype).flatten().tolist()
    
    # reference result
    if N > 0:
        gen = CustomPhilox4x(seed, config=config)
        out_ref = [gen.random_raw()[0] for _ in out_tri]
        assert out_tri == out_ref
    else:
        assert len(out_tri) == 0 # If N is 0, out_tri should be empty


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


# def test_get_results():
#     print(result_gold)
######################################## HELPERS for Eval ########################################