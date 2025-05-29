import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'num_stages': 2, 'num_warps': 4}), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'num_stages': 2, 'num_warps': 4}), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 8})], key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
        b_ptrs = b_ptr + ((k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_start + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_start + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        c_acc += tl.dot(a, b)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c_acc, mask=mask)
def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Pythonic interface for the Triton matmul kernel.
    """
    assert a.is_cuda and b.is_cuda, 'Input tensors must be on GPU.'
    assert a.ndim == 2 and b.ndim == 2, 'Only 2D tensors are supported.'
    (M, K) = a.shape
    (K_b, N) = b.shape
    assert K == K_b, f'Incompatible dimensions for matmul: {K} vs {K_b}.'
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32

    def grid(meta):
        return ((M + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'], (N + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'])
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c
def test_triton_matmul():
    results = {}
    try:
        A = torch.randn((16, 16), device='cuda', dtype=torch.float32)
        B = torch.randn((16, 16), device='cuda', dtype=torch.float32)
        C = triton_matmul(A, B)
        ref = A @ B
        results['test_case_1'] = torch.allclose(C, ref, atol=0.0001)
    except Exception as e:
        results['test_case_1'] = f'Error: {str(e)}'
    try:
        A = torch.randn((32, 8), device='cuda', dtype=torch.float32)
        B = torch.randn((8, 64), device='cuda', dtype=torch.float32)
        C = triton_matmul(A, B)
        ref = A @ B
        results['test_case_2'] = torch.allclose(C, ref, atol=0.0001)
    except Exception as e:
        results['test_case_2'] = f'Error: {str(e)}'
    try:
        A = torch.randn((10, 20), device='cuda', dtype=torch.float32)
        B = torch.randn((21, 10), device='cuda', dtype=torch.float32)
        C = triton_matmul(A, B)
        results['test_case_3'] = 'Unexpected pass'
    except AssertionError as e:
        results['test_case_3'] = f'Expected dimension error: {str(e)}'
    except Exception as e:
        results['test_case_3'] = f'Some other error: {str(e)}'
    try:
        A = torch.randn((64, 64), device='cuda', dtype=torch.float32)
        B = torch.randn((64, 64), device='cuda', dtype=torch.float32)
        C = triton_matmul(A, B)
        ref = A @ B
        results['test_case_4'] = torch.allclose(C, ref, atol=0.0001)
    except Exception as e:
        results['test_case_4'] = f'Error: {str(e)}'
    print(results)
def grid(meta):
    return ((M + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'], (N + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'])
##################################################################################################################################################



import torch

# Function to compare results of Triton and PyTorch matmul
def test_matmul():
    results = {}
    
    # Test case 1
    M, K, N = 256, 256, 256
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c_triton_1 = triton_matmul(a, b)
    results['test_case_1'] = c_triton_1

    # Test case 2
    M, K, N = 64, 64, 64
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c_triton_2 = triton_matmul(a, b)
    results['test_case_2'] = c_triton_2

    # Test case 3
    M, K, N = 16, 16, 16
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c_triton_3 = triton_matmul(a, b)
    results['test_case_3'] = c_triton_3

    return results

# Run the comparison
result_gold = test_matmul()