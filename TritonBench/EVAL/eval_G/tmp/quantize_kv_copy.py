import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(K_ptr, DestLoc_ptr, Out_ptr, OutScale_ptr, B, H, D, group_size, BLOCK_GROUP_NUM, BLOCK_GROUP_DIM, **meta):
    """
    Triton kernel for copying and quantizing K via destination index.
    Each block operates over a portion of the dimension D for one (batch, head) pair.
    1) Load data from K for each group.
    2) Calculate maximum absolute value within each group.
    3) Determine scale factor = max_abs_val / 127.0 (to fit int8 range).
    4) Quantize and store into Out according to destination indices from DestLoc.
    5) Save scale factors into OutScale for dequantization later.
    """
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    K_offset_base = b_idx * H * D + h_idx * D
    DestLoc_offset_base = b_idx * meta['seq_len']
    Out_offset_base = b_idx * H * D + h_idx * D
    OutScale_offset_base = b_idx * H * (D // group_size) + h_idx * (D // group_size)
    group_start = tl.program_id(2) * BLOCK_GROUP_NUM
    group_end = group_start + BLOCK_GROUP_NUM
    group_range = tl.arange(0, BLOCK_GROUP_NUM)
    group_ids = group_start + group_range
    for group_id in group_ids:
        if group_id >= D // group_size:
            break
        group_offset = group_id * group_size
        idxs_in_group = tl.arange(0, group_size)
        dim_positions = group_offset + idxs_in_group
        mask = dim_positions < D
        K_ptrs = K_ptr + (K_offset_base + dim_positions)
        data = tl.load(K_ptrs, mask=mask, other=0.0)
        abs_data = tl.abs(data)
        max_val = tl.maximum(abs_data[0], abs_data[1]) if group_size > 1 else abs_data[0]
        for i in range(2, group_size):
            if i < group_size:
                max_val = tl.maximum(max_val, abs_data[i])
        scale_factor = max_val / 127.0
        scale_factor = tl.maximum(scale_factor, 1e-08)
        quant_data = data / scale_factor
        quant_data_rounded = tl.round(quant_data)
        quant_data_clamped = tl.clip(quant_data_rounded, -128, 127)
        quant_data_int8 = quant_data_clamped.to(tl.int8)
        seq_offset = h_idx * D + group_offset
        seq_idx = meta['seq_idx']
        dest_loc_addr = DestLoc_ptr + DestLoc_offset_base + seq_idx
        dest_loc_val = tl.load(dest_loc_addr)
        out_write_offset = b_idx * H * D + dest_loc_val * D + group_offset
        out_ptrs = Out_ptr + out_write_offset + idxs_in_group
        tl.store(out_ptrs, quant_data_int8, mask=mask)
        scale_addr = OutScale_ptr + OutScale_offset_base + group_id
        tl.store(scale_addr, scale_factor)
def destindex_copy_quantize_kv(K: torch.Tensor, DestLoc: torch.Tensor, Out: torch.Tensor, OutScale: torch.Tensor, seq_len: int, seq_idx: int, group_size: int=8, BLOCK_GROUP_NUM: int=1, BLOCK_GROUP_DIM: int=32):
    """
    High-level wrapper to call `_fwd_kernel_destindex_copy_quantize_kv`.
    1) Validate input shapes and types.
    2) Compute grid size -> (batch, heads, groups).
    3) Launch the Triton kernel with the correct parameters.
    """
    assert K.is_cuda, 'K must be a CUDA tensor'
    assert DestLoc.is_cuda, 'DestLoc must be a CUDA tensor'
    assert Out.is_cuda, 'Out must be a CUDA tensor'
    assert OutScale.is_cuda, 'OutScale must be a CUDA tensor'
    (B, H, D) = K.shape
    assert D % group_size == 0, 'D must be divisible by group_size'
    grid = (B, H, (D // group_size + BLOCK_GROUP_NUM - 1) // BLOCK_GROUP_NUM)
    _fwd_kernel_destindex_copy_quantize_kv[grid](K, DestLoc, Out, OutScale, B, H, D, group_size, BLOCK_GROUP_NUM, BLOCK_GROUP_DIM, seq_len=seq_len, seq_idx=seq_idx, max_group_size=group_size, num_warps=4)
def test_destindex_copy_quantize_kv():
    """
    Test function for the quantization operator. All branches must be in this function.
    No parameters allowed. We store results in a dictionary with keys 'test_case_n'.
    """
    results = {}
    (B, H, D) = (1, 1, 8)
    group_size = 4
    K1 = torch.randn((B, H, D), device='cuda', dtype=torch.float32)
    DestLoc1 = torch.zeros((B, 16), device='cuda', dtype=torch.int32)
    DestLoc1[0, 0] = 0
    Out1 = torch.zeros_like(K1, dtype=torch.int8)
    OutScale1 = torch.zeros((B, H, D // group_size), device='cuda', dtype=torch.float32)
    seq_len_1 = 1
    seq_idx_1 = 0
    destindex_copy_quantize_kv(K1, DestLoc1, Out1, OutScale1, seq_len_1, seq_idx_1, group_size=group_size)
    results['test_case_1'] = (K1.clone().cpu(), Out1.clone().cpu(), OutScale1.clone().cpu())
    (B2, H2, D2) = (2, 2, 16)
    K2 = torch.randn((B2, H2, D2), device='cuda', dtype=torch.float32)
    DestLoc2 = torch.zeros((B2, 32), device='cuda', dtype=torch.int32)
    DestLoc2[0, 1] = 1
    DestLoc2[1, 1] = 2
    Out2 = torch.zeros_like(K2, dtype=torch.int8)
    OutScale2 = torch.zeros((B2, H2, D2 // 8), device='cuda', dtype=torch.float32)
    seq_len_2 = 2
    seq_idx_2 = 1
    destindex_copy_quantize_kv(K2, DestLoc2, Out2, OutScale2, seq_len_2, seq_idx_2, group_size=8)
    results['test_case_2'] = (K2.clone().cpu(), Out2.clone().cpu(), OutScale2.clone().cpu())
    (B3, H3, D3) = (1, 1, 8)
    K3 = torch.randn((B3, H3, D3), device='cuda', dtype=torch.float32)
    DestLoc3 = torch.zeros((B3, 10), device='cuda', dtype=torch.int32)
    DestLoc3[0, 0] = 2
    Out3 = torch.zeros_like(K3, dtype=torch.int8)
    OutScale3 = torch.zeros((B3, H3, D3 // 8), device='cuda', dtype=torch.float32)
    seq_len_3 = 1
    seq_idx_3 = 0
    destindex_copy_quantize_kv(K3, DestLoc3, Out3, OutScale3, seq_len_3, seq_idx_3, group_size=8)
    results['test_case_3'] = (K3.clone().cpu(), Out3.clone().cpu(), OutScale3.clone().cpu())
    (B4, H4, D4) = (2, 3, 12)
    group_size_4 = 6
    K4 = torch.randn((B4, H4, D4), device='cuda', dtype=torch.float32)
    DestLoc4 = torch.zeros((B4, 20), device='cuda', dtype=torch.int32)
    DestLoc4[0, 0] = 3
    DestLoc4[1, 7] = 5
    Out4 = torch.zeros_like(K4, dtype=torch.int8)
    OutScale4 = torch.zeros((B4, H4, D4 // group_size_4), device='cuda', dtype=torch.float32)
    seq_len_4 = 8
    seq_idx_4 = 7
    destindex_copy_quantize_kv(K4, DestLoc4, Out4, OutScale4, seq_len_4, seq_idx_4, group_size=group_size_4)
    results['test_case_4'] = (K4.clone().cpu(), Out4.clone().cpu(), OutScale4.clone().cpu())
    return results
##################################################################################################################################################



import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K,
    Dest_loc,
    Out,
    Out_scale,
    stride_k_bs,
    stride_k_h,
    stride_k_g,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_g,
    stride_o_d,
    stride_os_bs,
    stride_os_h,
    stride_os_g,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)

    dest_index = tl.load(Dest_loc + cur_index)

    src_data = tl.load(
        K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :],
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.0).to(Out_scale.dtype.element_ty)
    q_src_data = (src_data / data_scale[:, None]).to(tl.int8)

    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)
    return


@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    quant_group_dim = 8

    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy quant kv"
    grid = (seq_len, head_num)
    num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    Out = Out.view(Out.shape[0], Out.shape[1], group_size, group_dim)

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K,
        DestLoc,
        Out,
        Out_scale,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        Out_scale.stride(0),
        Out_scale.stride(1),
        Out_scale.stride(2),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


#######################################################################################################


import torch

def test_destindex_copy_quantize_kv():
    # Define the input tensors
    batch_size = 2
    head_num = 4
    head_dim = 16
    seq_len = 10
    quant_group_dim = 8

    # Ensure head_dim is divisible by quant_group_dim
    assert head_dim % quant_group_dim == 0

    # Create random input tensors
    K = torch.randn((seq_len, head_num, head_dim), dtype=torch.float32, device='cuda')
    DestLoc = torch.randint(0, seq_len, (seq_len,), dtype=torch.int32, device='cuda')
    Out = torch.empty_like(K, dtype=torch.int8)
    Out_scale = torch.empty((seq_len, head_num, head_dim // quant_group_dim), dtype=torch.float32, device='cuda')

    # Case 1: Normal execution (no early exit conditions)
    destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale)
    result_case_1 = {
        "Out": Out,
        "Out_scale": Out_scale
    }

    # Case 2: Handle a small batch size, less than group_size
    batch_size_small = 1
    K_small = torch.randn((batch_size_small, head_num, head_dim), dtype=torch.float32, device='cuda')
    DestLoc_small = torch.randint(0, seq_len, (batch_size_small,), dtype=torch.int32, device='cuda')
    Out_small = torch.empty_like(K_small, dtype=torch.int8)
    Out_scale_small = torch.empty((batch_size_small, head_num, head_dim // quant_group_dim), dtype=torch.float32, device='cuda')

    destindex_copy_quantize_kv(K_small, DestLoc_small, Out_small, Out_scale_small)
    result_case_2 = {
        "Out": Out_small,
        "Out_scale": Out_scale_small
    }

    # Case 3: Modify DestLoc to contain different sequence lengths
    DestLoc_varied = torch.randint(0, seq_len, (seq_len // 2,), dtype=torch.int32, device='cuda')
    Out_varied = torch.empty_like(K, dtype=torch.int8)
    Out_scale_varied = torch.empty((seq_len // 2, head_num, head_dim // quant_group_dim), dtype=torch.float32, device='cuda')

    destindex_copy_quantize_kv(K, DestLoc_varied, Out_varied, Out_scale_varied)
    result_case_3 = {
        "Out": Out_varied,
        "Out_scale": Out_scale_varied
    }

    # Case 4: Head dimension not divisible by quant_group_dim (assert will trigger)
    try:
        head_dim_invalid = 15  # Invalid head_dim
        K_invalid = torch.randn((seq_len, head_num, head_dim_invalid), dtype=torch.float32, device='cuda')
        DestLoc_invalid = torch.randint(0, seq_len, (seq_len,), dtype=torch.int32, device='cuda')
        Out_invalid = torch.empty_like(K_invalid, dtype=torch.int8)
        Out_scale_invalid = torch.empty((seq_len, head_num, head_dim_invalid // quant_group_dim), dtype=torch.float32, device='cuda')

        destindex_copy_quantize_kv(K_invalid, DestLoc_invalid, Out_invalid, Out_scale_invalid)
    except AssertionError as e:
        result_case_4 = str(e)

    return {
        "result_case_1": result_case_1,
        "result_case_2": result_case_2,
        "result_case_3": result_case_3,
        "result_case_4": result_case_4,
    }

result_gold = test_destindex_copy_quantize_kv()
