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


@triton.jit
def load_checked_block(block_ptr, IS_DIVISIBLE: tl.constexpr, SAFE_HEAD_DIM: tl.constexpr):
    if IS_DIVISIBLE and SAFE_HEAD_DIM:
        return tl.load(block_ptr)
    elif IS_DIVISIBLE and not SAFE_HEAD_DIM:
        return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
    elif not IS_DIVISIBLE and SAFE_HEAD_DIM:
        return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
    else:
        return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")


@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices


@triton.jit
def get_offset_for_next_block(
    loop_iter, col_indices, total_blocks,
    SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset


@triton.jit
def forward_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
    q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
    acc, l_i, m_i,
    off_z, off_h, offs_m, offs_n,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    PRESCALE_QK: tl.constexpr = False
    ROWS_GUARANTEED_SAFE: tl.constexpr = False
    IS_DIVISIBLE: tl.constexpr = True
    SM_SCALE: tl.constexpr = 0.08838834764831843
    SAFE_HEAD_DIM: tl.constexpr = True
    
    k = load_checked_block(K_block_ptr, SAFE_HEAD_DIM, IS_DIVISIBLE)
    qk = tl.dot(q, k, input_precision='ieee')
    if not PRESCALE_QK:
        qk *= SM_SCALE
    
    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)
    
    post_mod_scores = qk
    
    if CHECK_BLOCK_BOUNDARY:
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))
    
    if not IS_FULL_BLOCKS:
        # Block sparse mask logic from original Inductor code
        tmp1 = tl.full([1], False, tl.int1)
        tmp2 = tl.full([1], True, tl.int1)
        tmp3 = m
        tmp4 = tl.full([1], 0, tl.int32)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1], 4352, tl.int32)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp5 & tmp7
        tmp9 = tmp2 & tmp8
        tmp10 = tl.full([1], 256, tl.int32)
        tmp11 = tl.where((tmp3 < 0) != (tmp10 < 0), tl.where(tmp3 % tmp10 != 0, tmp3 // tmp10 - 1, tmp3 // tmp10), tmp3 // tmp10)
        tmp12 = n
        tmp13 = tl.where((tmp12 < 0) != (tmp10 < 0), tl.where(tmp12 % tmp10 != 0, tmp12 // tmp10 - 1, tmp12 // tmp10), tmp12 // tmp10)
        tmp14 = tmp13 + tmp4
        tmp15 = tmp11 - tmp14
        tmp16 = tl.abs(tmp15)
        tmp17 = tl.full([1], 1, tl.int32)
        tmp18 = tmp16 < tmp17
        tmp19 = tmp11 >= tmp14
        tmp20 = tmp18 & tmp19
        tmp21 = tmp9 & tmp20
        tmp22 = tmp1 | tmp21
        mask_mod_output = tmp22
        
        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, False)
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    
    m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij
    
    alpha = tl.math.exp2(m_i - m_ij_masked)
    p = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])
    
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]
    v = load_checked_block(V_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision='ieee')
    
    m_i = m_ij
    
    return acc, l_i, m_i


@triton.jit
def forward_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
    q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
    acc, l_i, m_i,
    off_z, off_h, offs_m, offs_n,
    kv_indices, kv_num_blocks,
    block_n_start, block_n_end,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    PRESCALE_QK: tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr = False
    IS_DIVISIBLE: tl.constexpr = True
    SM_SCALE: tl.constexpr = 0.08838834764831843
    SPARSE_KV_BLOCK_SIZE: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 128
    
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    RCP_LN2: tl.constexpr = 1.44269504
    
    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)
    
    for start_n in range(block_n_start, block_n_end):
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
                q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                acc, l_i, m_i,
                off_z, off_h, offs_m, offs_n,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS,
            )
        else:
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
                q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                acc, l_i, m_i,
                off_z, off_h, offs_m, offs_n,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )
        
        offset = get_offset_for_next_block(
            start_n, kv_indices, kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N, BLOCKS_ARE_CONTIGUOUS
        )
        
        V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, offset))
        offs_n = offs_n + offset
    
    return acc, l_i, m_i


@triton.jit
def block_sparse_flash_attention_kernel(arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0):
    PRESCALE_QK: tl.constexpr = False
    OUTPUT_LOGSUMEXP: tl.constexpr = True
    IS_DIVISIBLE: tl.constexpr = True
    SM_SCALE: tl.constexpr = 0.08838834764831843
    GQA_SHARED_HEADS: tl.constexpr = 1
    HAS_FULL_BLOCKS: tl.constexpr = True
    QK_HEAD_DIM: tl.constexpr = 128
    QK_HEAD_DIM_ROUNDED: tl.constexpr = 128
    V_HEAD_DIM: tl.constexpr = 128
    V_HEAD_DIM_ROUNDED: tl.constexpr = 128
    SAFE_HEAD_DIM: tl.constexpr = True
    kpack: tl.constexpr = 2
    BLOCK_M: tl.constexpr = 128
    BLOCK_N: tl.constexpr = 128
    SPARSE_Q_BLOCK_SIZE: tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE: tl.constexpr = 128
    
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE
    KV_NUM_BLKS = arg_KV_NUM_BLKS
    KV_IDX = arg_KV_IDX
    FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
    FULL_KV_IDX = arg_FULL_KV_IDX
    
    stride_qz, stride_qh, stride_qm, stride_qk = 16711680, 128, 3840, 1
    stride_kz, stride_kh, stride_kn, stride_kk = 16711680, 128, 3840, 1
    stride_vz, stride_vh, stride_vn, stride_vk = 16711680, 128, 3840, 1
    
    ZQ = 4
    HQ = 30
    Q_LEN = 4352
    ZKV = 4
    KV_LEN = 4352
    
    MATMUL_PRECISION = Q.dtype.element_ty
    
    q_start = tl.program_id(0)
    off_zq = tl.program_id(1) // HQ
    off_hq = tl.program_id(1) % HQ
    
    off_zkv = off_zq % ZKV
    off_hkv = off_hq // GQA_SHARED_HEADS
    
    q_offset = off_zq * stride_qz + off_hq * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh
    
    Q = Q + q_offset
    K = K + k_offset
    V = V + v_offset
    
    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    
    stride_kv_num_blks_h = 34
    stride_kv_idx_h = 1156
    stride_kv_idx_m = 34
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)
    
    offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
    
    sparse_hz_offset = off_zq + off_hq
    sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + q_start // SPARSE_Q_MULTIPLE
    sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + (q_start // SPARSE_Q_MULTIPLE) * stride_kv_idx_m
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Q_LEN, QK_HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(q_start * BLOCK_M, 0),
        block_shape=(BLOCK_M, QK_HEAD_DIM_ROUNDED),
        order=(1, 0)
    )
    q = load_checked_block(Q_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    
    kv_indices = KV_IDX + sparse_kv_idx_offset
    kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
    block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
    
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(QK_HEAD_DIM, KV_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, kv_start),
        block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(KV_LEN, V_HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(kv_start, 0),
        block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
        order=(1, 0)
    )
    offs_n = kv_start + tl.arange(0, BLOCK_N)
    
    acc, l_i, m_i = forward_inner(
        arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
        q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
        acc, l_i, m_i,
        off_zq, off_hq, offs_m[:, None], offs_n[None, :],
        kv_indices, kv_num_blocks,
        0, block_n_end,
        MATMUL_PRECISION,
        IS_FULL_BLOCKS=False,
    )
    
    if HAS_FULL_BLOCKS:
        kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
        
        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(QK_HEAD_DIM, KV_LEN),
            strides=(stride_kk, stride_kn),
            offsets=(0, kv_start),
            block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
            order=(0, 1)
        )
        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(KV_LEN, V_HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(kv_start, 0),
            block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
            order=(1, 0)
        )
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        
        acc, l_i, m_i = forward_inner(
            arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
            q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
            acc, l_i, m_i,
            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
            kv_indices, kv_num_blocks,
            0, block_n_end,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=True,
        )
    
    l_i = tl.where(l_i == 0.0, 1, l_i)
    acc = acc / l_i[:, None]
    
    idx_zq = tl.program_id(1) // HQ
    idx_hq = tl.program_id(1) % HQ
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)[None, :]
    
    mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)
    
    tl.store(out_ptr0 + (tl.broadcast_to(idx_d + 128*idx_hq + 3840*idx_m + 16711680*idx_zq, acc.shape)), acc, mask)
    
    if OUTPUT_LOGSUMEXP:
        off_hz = tl.program_id(1)
        l_ptrs = LSE + off_hz * Q_LEN + offs_m
        lse = m_i + tl.math.log2(l_i)
        if IS_DIVISIBLE:
            tl.store(l_ptrs, lse)
        else:
            tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)

##################################################################################################################################################

import numpy as np
import random
import torch 
import os
from numpy.random import RandomState
import pytest
from torch.testing import assert_close
from geak_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
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
# Helper function to define GB/s for block sparse flash attention
def calculate_block_sparse_flash_attention_gbps(params: Dict, ms: float) -> float:
    batch = params['batch']
    heads = params['heads']
    seq_len = params['seq_len']
    head_dim = params['head_dim']
    dtype = dtype_mapping[params['dtype_str']]
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    # Read Q, K, V; write Out and LSE
    # Approximate for block sparse (actual may be less due to sparsity)
    total_elements = batch * heads * seq_len * head_dim * 4  # Q, K, V, Out
    total_elements += batch * heads * seq_len  # LSE (float32, but approximate)
    total_bytes = total_elements * bytes_per_element
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

# Helper function to define TFLOPS for block sparse flash attention
def calculate_block_sparse_flash_attention_tflops(params: Dict, ms: float) -> float:
    batch = params['batch']
    heads = params['heads']
    seq_len = params['seq_len']
    head_dim = params['head_dim']
    # Q@K^T: batch * heads * seq_len * seq_len * head_dim * 2
    # P@V: batch * heads * seq_len * seq_len * head_dim * 2
    # Block sparse reduces this, but use full as approximation
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
                         [(4, 30, 4352, 128, 'bfloat16')])
def test_block_sparse_flash_attention(batch, heads, seq_len, head_dim, dtype_str, request):
    """
    Test block sparse Flash Attention using EXACT configuration from PyTorch Inductor
    Original: model__2_forward_5.2_output_code.py
    """
    set_seed()
    
    dtype = dtype_mapping[dtype_str]
    
    # Import rand_strided to match original Inductor's data generation EXACTLY
    from torch._dynamo.testing import rand_strided
    
    # Create inputs with EXACT strides from original (16711680, 128, 3840, 1)
    # Using rand_strided exactly as the original PyTorch Inductor code does
    Q = rand_strided((batch, heads, seq_len, head_dim), (16711680, 128, 3840, 1), device='cuda', dtype=dtype)
    K = rand_strided((batch, heads, seq_len, head_dim), (16711680, 128, 3840, 1), device='cuda', dtype=dtype)
    V = rand_strided((batch, heads, seq_len, head_dim), (16711680, 128, 3840, 1), device='cuda', dtype=dtype)
    
    # Create block sparse mask tensors with EXACT shapes and strides from original
    # Using rand_strided with exact same parameters as original Inductor benchmark
    # KV_IDX (primals_4): (1, 1, 34, 34), stride (1156, 1156, 34, 1)
    kv_idx = rand_strided((1, 1, 34, 34), (1156, 1156, 34, 1), device='cuda', dtype=torch.int32)
    # KV_NUM_BLKS (primals_5): (1, 1, 34), stride (34, 34, 1)
    kv_num_blks = rand_strided((1, 1, 34), (34, 34, 1), device='cuda', dtype=torch.int32)
    # FULL_KV_NUM_BLKS (primals_6): (1, 1, 34), stride (34, 34, 1)
    full_kv_num_blks = rand_strided((1, 1, 34), (34, 34, 1), device='cuda', dtype=torch.int32)
    # FULL_KV_IDX (primals_7): (1, 1, 34, 34), stride (1156, 1156, 34, 1)
    full_kv_idx = rand_strided((1, 1, 34, 34), (1156, 1156, 34, 1), device='cuda', dtype=torch.int32)
    
    # Create outputs with EXACT shapes and strides from original
    # LSE (buf0): (4, 30, 4352), stride (130560, 4352, 1)
    LSE = torch.empty((batch, heads, seq_len), device='cuda', dtype=torch.float32)
    # Out (buf1): (4, 30, 4352, 128), stride (16711680, 128, 3840, 1)
    Out = torch.empty_like(Q)
    
    # Launch kernel with EXACT grid from original: (34, 120, 1)
    # CRITICAL: Match original's num_warps=4, num_stages=1 for performance
    grid = (34, 120, 1)
    block_sparse_flash_attention_kernel[grid](
        Q, K, V, LSE, kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx, Out,
        num_warps=4, num_stages=1
    )
    
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = Out.clone().detach().cpu()
    ################################################################### 
    
    # Basic sanity checks
    assert not torch.isnan(Out).any(), "Output contains NaN"
    assert Out.shape == Q.shape, f"Shape mismatch: {Out.shape} vs {Q.shape}"
    assert LSE.shape == (batch, heads, seq_len), f"LSE shape mismatch: {LSE.shape}"


OP_NAME_FOR_BENCHMARK = "block_sparse_flash_attention_perf"

@pytest.mark.parametrize('batch,heads,seq_len,head_dim,dtype_str',
                         [(4, 30, 4352, 128, 'bfloat16')])
def test_performance(batch, heads, seq_len, head_dim, dtype_str, request):
    """
    Benchmark block sparse Flash Attention using EXACT configuration from PyTorch Inductor
    Original: model__2_forward_5.2_output_code.py benchmark_compiled_module()
    Uses rand_strided to match original's data generation for accurate performance comparison
    """
    set_seed()
    dtype = dtype_mapping[dtype_str]
    
    # Import rand_strided to match original Inductor's data generation EXACTLY
    from torch._dynamo.testing import rand_strided
    
    # Create inputs with EXACT strides from original (matching benchmark_compiled_module)
    Q = rand_strided((batch, heads, seq_len, head_dim), (16711680, 128, 3840, 1), device='cuda', dtype=dtype)
    K = rand_strided((batch, heads, seq_len, head_dim), (16711680, 128, 3840, 1), device='cuda', dtype=dtype)
    V = rand_strided((batch, heads, seq_len, head_dim), (16711680, 128, 3840, 1), device='cuda', dtype=dtype)
    
    # Block sparse mask tensors - using rand_strided with exact strides from original
    kv_idx = rand_strided((1, 1, 34, 34), (1156, 1156, 34, 1), device='cuda', dtype=torch.int32)
    kv_num_blks = rand_strided((1, 1, 34), (34, 34, 1), device='cuda', dtype=torch.int32)
    full_kv_num_blks = rand_strided((1, 1, 34), (34, 34, 1), device='cuda', dtype=torch.int32)
    full_kv_idx = rand_strided((1, 1, 34, 34), (1156, 1156, 34, 1), device='cuda', dtype=torch.int32)
    
    # Outputs
    LSE = torch.empty((batch, heads, seq_len), device='cuda', dtype=torch.float32)
    Out = torch.empty_like(Q)
    
    def op_lambda():
        grid = (34, 120, 1)  # EXACT grid from original
        # CRITICAL: Match original's num_warps=4, num_stages=1
        block_sparse_flash_attention_kernel[grid](
            Q, K, V, LSE, kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx, Out,
            num_warps=4, num_stages=1
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
                              gbps_calculator=calculate_block_sparse_flash_attention_gbps,
                              tflops_calculator=calculate_block_sparse_flash_attention_tflops)

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
