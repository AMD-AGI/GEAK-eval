# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

dtype_mapping = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
}
######################################## Imports ######################################## 


@triton.autotune(
    configs=[
        # Small blocks - high parallelism
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=2),
        
        # Medium balanced blocks
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        
        # Larger blocks
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def type_conversion_transpose_kernel_optimized(
    in_ptr,
    out_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Optimized type conversion with transpose: fp32 → bf16, (M x N) → (N x M)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Boundary masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load from input (M x N)
    in_idx = offs_m[:, None] * N + offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    
    x = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    
    # Convert to bf16
    x_bf16 = x.to(tl.bfloat16)
    
    # Store to output with transpose (N x M)
    out_idx = offs_n[:, None] * M + offs_m[None, :]
    mask_t = mask_n[:, None] & mask_m[None, :]
    
    tl.store(out_ptr + out_idx, tl.trans(x_bf16), mask=mask_t)


##################################################################################################################################################

import numpy as np
import random
import torch 
import os
from numpy.random import RandomState
import pytest
from torch.testing import assert_close
from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict

import triton
import triton.language as tl

dtype_mapping = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
}

result_gold = {}

######################################## HELPERS for Eval ######################################## 
# Helper function to define GB/s for type_conversion_transpose
def calculate_type_conversion_transpose_gbps(params: Dict, ms: float) -> float:
    M = params['M']
    N = params['N']
    size = M * N
    # Read fp32 (4 bytes), write bf16 (2 bytes)
    total_bytes = size * 4 + size * 2
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

# Helper function to define TFLOPS for type_conversion_transpose
def calculate_type_conversion_transpose_tflops(params: Dict, ms: float) -> float:
    # Type conversion + transpose is not a compute operation
    return 0.0

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


@pytest.mark.parametrize('M,N',
                         [(3840, 3840)])
def test_type_conversion_transpose(M, N, request):
    set_seed()

    in_tensor = torch.randn(M, N, device='cuda', dtype=torch.float32)
    out_tensor = torch.empty(N, M, device='cuda', dtype=torch.bfloat16)
    
    expected = in_tensor.to(torch.bfloat16).t()

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
    
    type_conversion_transpose_kernel_optimized[grid](
        in_tensor,
        out_tensor,
        M, N,
    )

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = out_tensor.clone().detach().cpu()
    ################################################################### 

    assert_close(out_tensor.float(), expected.float(), rtol=1e-2, atol=1e-2, check_dtype=False)


OP_NAME_FOR_BENCHMARK = "type_conversion_transpose_kernel_perf"

@pytest.mark.parametrize('M,N',
                         [(3840, 3840),
                          (2048, 2048)])
def test_performance(M, N, request):
    set_seed()
    in_tensor = torch.randn(M, N, device='cuda', dtype=torch.float32)
    out_tensor = torch.empty(N, M, device='cuda', dtype=torch.bfloat16)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    kernel_args = [in_tensor, out_tensor, M, N]
    
    op_lambda = lambda: type_conversion_transpose_kernel_optimized[grid](*kernel_args)

    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_calculators = {"M": M, "N": N}

    benchmarker.run_benchmark(current_params_dict=current_params_for_calculators,
                              gbps_calculator=calculate_type_conversion_transpose_gbps,
                              tflops_calculator=calculate_type_conversion_transpose_tflops)

######################################## HELPERS for Eval ########################################     
# --- Pytest hook to save the dictionary at the end of the session ---  
def test_save_results():  
    """  
    Called after whole test run finished, right before returning the exit status to the system.  
    """
    print('Inside session finish...')
    if "_CALL_SUCCESS_" not in result_gold:
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[0.0]])
    # FIX: Only replace the .py extension, not all dots in the path
    OUTPUT_FILENAME = __file__[:-3] + '_py.pt'
    print(f"\nSaving all y_triton results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    try:
        torch.save(result_gold, OUTPUT_FILENAME)       
        print(f"✓ Successfully saved {len(result_gold)} y_triton tensors to {OUTPUT_FILENAME}.")
    except Exception as e:
        print(f"✗ SAVE FAILED: {e}")
        print(f"✗ Attempted path: {OUTPUT_FILENAME}")
        print(f"✗ Current __file__: {__file__}")
        raise  


def test_save_performance_results():
    """
    Called after the test_performance function finishes.
    This is a separate hook to ensure performance results are saved.
    """
    print('\nPytest session finishing... Saving benchmark results...')

    output_directory = os.path.join(os.path.dirname(__file__), "perf")  # Save in a "perf" subdirectory next to the test file
    os.makedirs(output_directory, exist_ok=True)
    
    save_all_benchmark_results(output_directory)
    # import pdb; pdb.set_trace()
    print(f"All benchmark results attempted to save to: {output_directory}")


######################################## HELPERS for Eval ########################################
