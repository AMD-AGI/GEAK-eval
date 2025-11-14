# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import triton  
import triton.language as tl  


@triton.jit  
def moe_gemm_kernel(  
    A,  
    B,  
    Out,  
    A_scale,  
    B_scale,  
    stride_am,  
    stride_ak,  
    stride_be,  
    stride_bn,  
    stride_bk,  
    stride_cm,  
    stride_cn,  
    stride_bse,  
    stride_bsn,  
    top_k: tl.constexpr,  
    topk_weights_ptr,  
    sorted_token_ids_ptr,  
    expert_ids_ptr,  
    EM: tl.constexpr,  
    N: tl.constexpr,  
    K: tl.constexpr,  
    EVEN_K: tl.constexpr,  
    MUL_ROUTED_WEIGHT: tl.constexpr,  
    use_fp8_w8a8: tl.constexpr,  
    use_int8_w8a16: tl.constexpr,  
    use_int8_w8a8: tl.constexpr,  
    BLOCK_SIZE_M: tl.constexpr,  
    BLOCK_SIZE_N: tl.constexpr,  
    BLOCK_SIZE_K: tl.constexpr,  
    GROUP_SIZE_M: tl.constexpr,  
):  
    """  
    Implements the fused computation for a Mixture of Experts (MOE) using  
    token and expert matrices.  
  
    Key Parameters:  
    - A: The input tensor representing tokens with shape (*, K), where '*' can  
        be any shape representing batches and K is the feature dimension of  
        each token.  
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is  
        the number of experts, K is the input feature dimension, and N is  
        the output feature dimension.  
    - C: The output cache tensor with shape (M, topk, N), where M is the  
        total number of tokens post padding, topk is the number of times  
        each token is repeated, and N is the output feature dimension.  
    - sorted_token_ids: A tensor containing the sorted indices of tokens,  
        repeated topk times and arranged by the expert index they are  
        assigned to.  
    - expert_ids: A tensor containing the indices of the expert for each  
        block. It determines which expert matrix from B should be used for  
        each block in A.  
    This kernel performs the multiplication of a token by its corresponding  
    expert matrix as determined by `expert_ids`. The sorting of  
    `sorted_token_ids` by expert index and padding ensures divisibility by  
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix  
    multiplication across different blocks processed by the same expert.  
    """  
    pid = tl.program_id(axis=0)  
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)  
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  
    group_id = pid // num_pid_in_group  
    first_pid_m = group_id * GROUP_SIZE_M  
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)  
    pid_n = (pid % num_pid_in_group) // group_size_m  
  
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)  
  
    # Here we assume that valid tokens are in the range [0, M).  
    token_mask = (offs_token >= 0) & (offs_token < EM)  
  
    off_experts = tl.load(expert_ids_ptr + pid_m)  
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N  
    offs_k = tl.arange(0, BLOCK_SIZE_K)  
    a_ptrs = A + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)  
    b_ptrs = B + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)  
  
    if use_int8_w8a16:  
        b_scale_ptrs = B_scale + off_experts * stride_bse + offs_bn[None, :] * stride_bsn  
        b_scale = tl.load(b_scale_ptrs)  
  
    if use_fp8_w8a8 or use_int8_w8a8:  
        a_scale = tl.load(A_scale)  
        b_scale = tl.load(B_scale + off_experts)  
  
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  
  
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  
        # Masking ensures we don't load from invalid tokens or indices  
        if EVEN_K:  
            a = tl.load(a_ptrs, mask=(token_mask[:, None]), other=0.0)  
            b = tl.load(b_ptrs)  
        else:  
            a = tl.load(a_ptrs, mask=(token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)), other=0.0)  
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)  
  
        if use_int8_w8a16:  
            accumulator = tl.dot(a, b.to(a.dtype), acc=accumulator)  
        elif use_fp8_w8a8 or use_int8_w8a8:  
            accumulator += tl.dot(a, b)  
        else:  
            accumulator = tl.dot(a, b, acc=accumulator)  
  
        a_ptrs += BLOCK_SIZE_K * stride_ak  
        b_ptrs += BLOCK_SIZE_K * stride_bk  
  
    if MUL_ROUTED_WEIGHT:  
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)  
        accumulator = accumulator * moe_weight[:, None]  
  
    if use_int8_w8a16:  
        accumulator = (accumulator * b_scale).to(Out.dtype.element_ty)  
    elif use_fp8_w8a8 or use_int8_w8a8:  
        accumulator = (accumulator * a_scale * b_scale).to(Out.dtype.element_ty)  
    else:  
        accumulator = accumulator.to(Out.dtype.element_ty)  
  
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  
    out_ptrs = Out + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]  
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)  
    tl.store(out_ptrs, accumulator.to(Out.dtype.element_ty), mask=c_mask)


