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
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        
        # Medium blocks - balanced
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        
        # Larger blocks
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        
        # High warp configs for AMD
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
    ],
    key=['seq_len', 'head_dim'],
)
@triton.jit
def flash_attention_forward_kernel_optimized(
    Q, K, V, sm_scale,
    Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Autotuned Flash Attention with simplified implementation
    """
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Compute batch and head indices
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    
    # Compute base offsets
    off_q = pid_b * stride_qb + pid_h * stride_qh
    off_k = pid_b * stride_kb + pid_h * stride_kh
    off_v = pid_b * stride_vb + pid_h * stride_vh
    off_o = pid_b * stride_ob + pid_h * stride_oh
    
    # Initialize offsets for M dimension
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Boundary check for M
    mask_m = offs_m < seq_len
    
    # Load Q block once (stays in registers)
    q_ptrs = Q + off_q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize statistics
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Loop over K, V in chunks - compiler will pipeline with num_stages
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        # Load K block
        k_ptrs = K + off_k + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Compute attention scores: Q @ K^T
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        # Apply causal mask
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # Load V block
        v_ptrs = V + off_v + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Update accumulator with in-place operation
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_i_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    out_ptrs = Out + off_o + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


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
# Helper function to define GB/s for flash_attention
def calculate_flash_attention_gbps(params: Dict, ms: float) -> float:
    batch = params['batch']
    heads = params['heads']
    seq_len = params['seq_len']
    head_dim = params['head_dim']
    dtype = dtype_mapping[params['dtype_str']]
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    # Read Q, K, V; write Out
    total_elements = batch * heads * seq_len * head_dim * 4
    total_bytes = total_elements * bytes_per_element
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

# Helper function to define TFLOPS for flash_attention
def calculate_flash_attention_tflops(params: Dict, ms: float) -> float:
    batch = params['batch']
    heads = params['heads']
    seq_len = params['seq_len']
    head_dim = params['head_dim']
    # Q@K^T: batch * heads * seq_len * seq_len * head_dim * 2
    # P@V: batch * heads * seq_len * seq_len * head_dim * 2
    flops = batch * heads * seq_len * seq_len * head_dim * 2 * 2
    tflops = flops / (ms / 1000) / 1e12
    return tflops

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


@pytest.mark.parametrize('batch,heads,seq_len,head_dim,dtype_str',
                         [(2, 8, 512, 64, 'bfloat16')])
def test_flash_attention(batch, heads, seq_len, head_dim, dtype_str, request):
    set_seed()

    dtype = dtype_mapping[dtype_str]
    Q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=dtype)
    Out = torch.empty_like(Q)
    
    scale = 1.0 / (head_dim ** 0.5)
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    def grid(meta):
        return (triton.cdiv(seq_len, meta['BLOCK_M']), batch * heads)
    
    flash_attention_forward_kernel_optimized[grid](
        Q, K, V, scale,
        Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        batch, heads, seq_len, head_dim,
        BLOCK_D=BLOCK_D,
    )

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = Out.clone().detach().cpu()
    ################################################################### 

    # Basic sanity checks - full correctness check would require PyTorch's attention
    assert not torch.isnan(Out).any(), "Output contains NaN"
    assert Out.shape == Q.shape, f"Shape mismatch: {Out.shape} vs {Q.shape}"


OP_NAME_FOR_BENCHMARK = "flash_attention_kernel_perf"

@pytest.mark.parametrize('batch,heads,seq_len,head_dim,dtype_str',
                         [(2, 8, 512, 64, 'bfloat16'),
                          (4, 16, 256, 64, 'bfloat16')])
def test_performance(batch, heads, seq_len, head_dim, dtype_str, request):
    set_seed()
    dtype = dtype_mapping[dtype_str]
    Q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=dtype)
    Out = torch.empty_like(Q)
    
    scale = 1.0 / (head_dim ** 0.5)
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    def grid(meta):
        return (triton.cdiv(seq_len, meta['BLOCK_M']), batch * heads)

    def op_lambda():
        flash_attention_forward_kernel_optimized[grid](
            Q, K, V, scale,
            Out,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            batch, heads, seq_len, head_dim,
            BLOCK_D=BLOCK_D,
        )

    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_calculators = {
        "batch": batch,
        "heads": heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "dtype_str": dtype_str
    }

    benchmarker.run_benchmark(current_params_dict=current_params_for_calculators,
                              gbps_calculator=calculate_flash_attention_gbps,
                              tflops_calculator=calculate_flash_attention_tflops)

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


def test_save_performance_results():
    """
    Called after the test_performance function finishes.
    This is a separate hook to ensure performance results are saved.
    """
    print('\nPytest session finishing... Saving benchmark results...')

    output_directory = os.path.join(os.path.dirname(__file__), "perf")  # Save in a "perf" subdirectory next to the test file
    os.makedirs(output_directory, exist_ok=True)
    
    save_all_benchmark_results(output_directory)
    print(f"All benchmark results attempted to save to: {output_directory}")


######################################## HELPERS for Eval ########################################

