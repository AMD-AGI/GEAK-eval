import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_block_start = pid_m * BLOCK_SIZE_M
    n_block_start = pid_n * BLOCK_SIZE_N
    offs_am = m_block_start + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        k_current = k_block_start + offs_k
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + k_current[None, :] * stride_ak)
        b_ptrs = b_ptr + (k_current[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (k_current[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_current[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    mask_c = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)
def matmul(a: torch.Tensor, b: torch.Tensor):
    """
    A wrapper function to multiply a and b using the Triton matmul_kernel.
    This function supports float16 and (experimental) float8 data, using
    a specialized blocked approach for efficient GPU computation.
    """
    assert a.is_cuda and b.is_cuda, 'Input tensors must be on GPU.'
    assert a.dim() == 2 and b.dim() == 2, 'Only 2D matrices supported.'
    assert a.size(1) == b.size(0), 'Incompatible matrix dimensions for multiplication.'
    (M, K) = a.shape
    (K_, N) = b.shape
    assert K == K_, 'Incompatible matrix shapes.'
    if a.dtype == torch.float16:
        BLOCK_SIZE_M = 64
        num_stages = 3
        num_warps = 4
    elif hasattr(torch, 'float8_e4m3') and (a.dtype == torch.float8_e4m3 or a.dtype == torch.float8_e5m2):
        BLOCK_SIZE_M = 32
        num_stages = 2
        num_warps = 2
    else:
        raise ValueError('Unsupported data type for this matmul kernel.')
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, num_stages=num_stages, num_warps=num_warps)
    return c
def test_matmul_branches():
    """
    Single test function covering up to four branch tests.
    The results of each test are stored in a dictionary: results_dict.
    """
    results_dict = {}
    a1 = torch.randn((64, 64), dtype=torch.float16, device='cuda')
    b1 = torch.randn((64, 64), dtype=torch.float16, device='cuda')
    c1 = matmul(a1, b1)
    ref1 = a1 @ b1
    results_dict['test_case_1'] = torch.allclose(c1, ref1.float(), atol=0.01, rtol=0.01)
    a2 = torch.randn((128, 32), dtype=torch.float16, device='cuda')
    b2 = torch.randn((32, 256), dtype=torch.float16, device='cuda')
    c2 = matmul(a2, b2)
    ref2 = a2 @ b2
    results_dict['test_case_2'] = torch.allclose(c2, ref2.float(), atol=0.01, rtol=0.01)
    if hasattr(torch, 'float8_e4m3'):
        a3 = torch.randint(-128, 127, (64, 32), device='cuda').to(torch.float8_e4m3)
        b3 = torch.randint(-128, 127, (32, 64), device='cuda').to(torch.float8_e4m3)
        c3 = matmul(a3, b3)
        results_dict['test_case_3'] = c3.shape == (64, 64)
        if hasattr(torch, 'float8_e5m2'):
            a4 = torch.randint(-128, 127, (32, 32), device='cuda').to(torch.float8_e5m2)
            b4 = torch.randint(-128, 127, (32, 32), device='cuda').to(torch.float8_e5m2)
            c4 = matmul(a4, b4)
            results_dict['test_case_4'] = c4.shape == (32, 32)
        else:
            results_dict['test_case_4'] = None
    else:
        results_dict['test_case_3'] = None
        results_dict['test_case_4'] = None
    for (k, v) in results_dict.items():
        print(f'{k}: {v}')
##################################################################################################################################################



import torch

# Test for matmul
def test_matmul():
    results = {}
    M, K, N = 256, 128, 256

    # Test case 1: torch.float16
    a = torch.randn((M, K), dtype=torch.float16, device='cuda')
    b = torch.randn((K, N), dtype=torch.float16, device='cuda')
    c = matmul(a, b)
    results['test_case_1'] = c

    return results

# Run all tests
result_gold = test_matmul()