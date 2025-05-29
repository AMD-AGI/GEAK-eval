import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8)], key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, ACTIVATION: tl.constexpr, ALPHA: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + rm[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + rn[None, :] * stride_bn
    accumulation = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        a_block = tl.load(a_ptrs, mask=(rm[:, None] < M) & (k_block_start + tl.arange(0, BLOCK_SIZE_K)[None, :] < K), other=0.0)
        b_block = tl.load(b_ptrs, mask=(k_block_start + tl.arange(0, BLOCK_SIZE_K)[:, None] < K) & (rn[None, :] < N), other=0.0)
        accumulation += tl.dot(a_block, b_block)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == 1:
        negative_mask = accumulation < 0
        accumulation = tl.where(negative_mask, accumulation * ALPHA, accumulation)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, accumulation, mask=mask_c)
def leaky_relu(x: torch.Tensor, alpha: float=0.01):
    """
    Apply leaky ReLU in-place on a PyTorch GPU tensor x.
    Returns a new tensor with the transformation applied element-wise.
    """
    return torch.where(x >= 0, x, alpha * x)
def matmul(a: torch.Tensor, b: torch.Tensor, activation: int=0, alpha: float=0.01) -> torch.Tensor:
    """
    Matrix multiplication wrapper function with optional leaky ReLU.
    Args:
        a (torch.Tensor): [M, K] matrix
        b (torch.Tensor): [K, N] matrix
        activation (int): 0 or 1 to specify if leaky ReLU is applied
        alpha (float): leaky ReLU negative slope
    Returns:
        c (torch.Tensor): [M, N] result of compute
    """
    assert a.shape[1] == b.shape[0], 'Incompatible dimensions for matmul.'
    assert a.is_cuda and b.is_cuda, 'Tensors must be on GPU.'
    (M, K) = a.shape
    (K_, N) = b.shape
    assert K == K_, 'K dimension mismatch'
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    a_strides = a.stride()
    b_strides = b.stride()
    c_strides = c.stride()
    grid = lambda META: ((M + META['BLOCK_SIZE_M'] - 1) // META['BLOCK_SIZE_M'], (N + META['BLOCK_SIZE_N'] - 1) // META['BLOCK_SIZE_N'])
    matmul_kernel[grid](a, b, c, M, N, K, a_strides[0], a_strides[1], b_strides[0], b_strides[1], c_strides[0], c_strides[1], ACTIVATION=activation, ALPHA=alpha)
    return c
def test_matmul():
    """
    Single test function containing several branches to validate the matmul operator with/without leaky ReLU.
    """
    results = {}
    A1 = torch.randn((16, 16), device='cuda')
    B1 = torch.randn((16, 16), device='cuda')
    C1 = matmul(A1, B1, activation=0)
    results['test_case_1'] = C1
    A2 = torch.randn((32, 32), device='cuda')
    B2 = torch.randn((32, 32), device='cuda')
    C2 = matmul(A2, B2, activation=1, alpha=0.05)
    results['test_case_2'] = C2
    A3 = torch.randn((32, 8), device='cuda')
    B3 = torch.randn((8, 16), device='cuda')
    C3 = matmul(A3, B3, activation=0)
    results['test_case_3'] = C3
    A4 = torch.randn((64, 64), device='cuda')
    B4 = torch.randn((64, 64), device='cuda')
    C4 = matmul(A4, B4, activation=1, alpha=0.01)
    results['test_case_4'] = C4
    return results
##################################################################################################################################################



import torch

# Test case 1: Basic matrix multiplication without activation
def test_matmul():
    results = {}
    
    # Test case 1: Basic matrix multiplication without activation
    M, K, N = 128, 64, 256
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = matmul(a, b)
    results['test_case_1'] = c

    # Test case 2: Matrix multiplication with Leaky ReLU activation
    M, K, N = 128, 64, 256
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = matmul(a, b, activation="leaky_relu")
    results['test_case_2'] = c

    # Test case 3: Different matrix sizes
    M, K, N = 256, 128, 512
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = matmul(a, b)
    results['test_case_3'] = c

    return results

# Run tests
result_gold = test_matmul()
