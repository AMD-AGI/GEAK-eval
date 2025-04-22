import torch
import triton
import triton.language as tl
from triton.runtime.autotuner import Config

def next_power_of_2(x):
    return 1 << (x - 1).bit_length()
@triton.jit
def _fwd_kernel_destindex_copy_kv(K_ptr, Out_ptr, Dest_loc_ptr, batch_size, head_num, depth_num, strideKB, strideKH, strideKD, strideOB, strideOH, strideOD, strideDest, seq_len, BLOCK_HEAD: tl.constexpr):
    idx = tl.program_id(0)
    b = tl.load(Dest_loc_ptr + idx * strideDest)
    offs_h = tl.arange(0, BLOCK_HEAD)
    k_ptrs = K_ptr + (b * strideKB + offs_h * strideKH)
    o_ptrs = Out_ptr + (b * strideOB + offs_h * strideOH)
    mask = offs_h < head_num
    k_data = tl.load(k_ptrs, mask=mask, other=0.0)
    tl.store(o_ptrs, k_data, mask=mask)
def destindex_copy_kv(K, Dest_loc, Out):
    """
    Wrapper function for _fwd_kernel_destindex_copy_kv
    Copies data from K to Out using indices in Dest_loc.

    :param K: (B, H, D) tensor on CUDA
    :param Dest_loc: (L,) long/int tensor on CUDA (contains indices in range [0, B))
    :param Out: (B, H, D) tensor on CUDA
    """
    assert K.is_cuda, 'K must be a CUDA tensor'
    assert Out.is_cuda, 'Out must be a CUDA tensor'
    assert Dest_loc.is_cuda, 'Dest_loc must be a CUDA tensor'
    (B, H, D) = K.shape
    assert Out.shape == (B, H, D), 'Out tensor shape must match K shape'
    seq_len = Dest_loc.shape[0]
    BLOCK_HEAD = next_power_of_2(H)
    strideKB = K.stride(0)
    strideKH = K.stride(1)
    strideKD = K.stride(2)
    strideOB = Out.stride(0)
    strideOH = Out.stride(1)
    strideOD = Out.stride(2)
    strideDest = Dest_loc.stride(0)
    grid = (seq_len,)
    _fwd_kernel_destindex_copy_kv[grid](K, Out, Dest_loc, B, H, D, strideKB, strideKH, strideKD, strideOB, strideOH, strideOD, strideDest, seq_len, BLOCK_HEAD=BLOCK_HEAD, num_warps=1, num_stages=1)
def test_destindex_copy_kv():
    """
    Test code for the GPU operator _fwd_kernel_destindex_copy_kv and its wrapper destindex_copy_kv.
    All branch tests in a single function. Branch results stored in a dictionary.
    """
    results = {}
    (B1, H1, D1) = (4, 8, 1)
    K1 = torch.arange(B1 * H1 * D1, device='cuda', dtype=torch.float32).reshape(B1, H1, D1)
    Dest_loc1 = torch.tensor([3, 1, 0, 2], device='cuda', dtype=torch.long)
    Out1 = torch.zeros_like(K1)
    destindex_copy_kv(K1, Dest_loc1, Out1)
    results['test_case_1'] = Out1.tolist()
    (B2, H2, D2) = (5, 1, 3)
    K2 = torch.randn(B2, H2, D2, device='cuda')
    Dest_loc2 = torch.tensor([4, 0, 2, 1, 3], device='cuda', dtype=torch.long)
    Out2 = torch.zeros_like(K2)
    destindex_copy_kv(K2, Dest_loc2, Out2)
    results['test_case_2'] = Out2.cpu().tolist()
    (B3, H3, D3) = (3, 10, 2)
    K3 = torch.randn(B3, H3, D3, device='cuda')
    Dest_loc3 = torch.tensor([0, 2, 1], device='cuda', dtype=torch.long)
    Out3 = torch.zeros_like(K3)
    destindex_copy_kv(K3, Dest_loc3, Out3)
    results['test_case_3'] = Out3.cpu().tolist()
    (B4, H4, D4) = (2, 6, 4)
    K4 = torch.randn(B4, H4, D4, device='cuda')
    Dest_loc4 = torch.tensor([0, 1], device='cuda', dtype=torch.long)
    Out4 = torch.zeros_like(K4)
    destindex_copy_kv(K4, Dest_loc4, Out4)
    results['test_case_4'] = Out4.cpu().tolist()
    for (key, value) in results.items():
        print(key, value)
##################################################################################################################################################



def test_destindex_copy_kv():
    B, N_CTX, H, D = 32, 1024, 12, 128
    dest = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32, device="cuda")

    destindex_copy_kv(src, dest_loc, dest)
    test_case = torch.allclose(src, dest, atol=1e-2, rtol=0)

    return {
        "test_case_1": test_case
    }

result_gold = test_destindex_copy_kv()