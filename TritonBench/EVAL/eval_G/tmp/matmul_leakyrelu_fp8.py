import torch
import triton
import triton.language as tl

@torch.jit.script
def leaky_relu(x: torch.Tensor, alpha: float=0.01) -> torch.Tensor:
    return torch.where(x >= 0, x, alpha * x)
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, USE_ACTIVATION: tl.constexpr, ALPHA: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_block_start = pid_m * BLOCK_SIZE_M
    n_block_start = pid_n * BLOCK_SIZE_N
    a_offset = m_block_start + tl.arange(0, BLOCK_SIZE_M)
    b_offset = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        a_k_offset = k_block_start + tl.arange(0, BLOCK_SIZE_K)
        b_k_offset = k_block_start + tl.arange(0, BLOCK_SIZE_K)
        A_block = tl.load(A_ptr + (a_offset[:, None] * stride_am + a_k_offset[None, :] * stride_ak), mask=(a_offset[:, None] < M) & (a_k_offset[None, :] < K), other=0.0)
        B_block = tl.load(B_ptr + (b_k_offset[:, None] * stride_bk + b_offset[None, :] * stride_bn), mask=(b_k_offset[:, None] < K) & (b_offset[None, :] < N), other=0.0)
        acc += tl.dot(A_block, B_block)
    if USE_ACTIVATION:
        acc = tl.where(acc >= 0, acc, ALPHA * acc)
    c_offset = m_block_start + tl.arange(0, BLOCK_SIZE_M)
    c_lane = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    mask = (c_offset[:, None] < M) & (c_lane[None, :] < N)
    tl.store(C_ptr + (c_offset[:, None] * stride_cm + c_lane[None, :] * stride_cn), acc, mask=mask)
def matmul(A: torch.Tensor, B: torch.Tensor, activation: str=None, alpha: float=0.01, BLOCK_SIZE_M: int=32, BLOCK_SIZE_N: int=32, BLOCK_SIZE_K: int=32) -> torch.Tensor:
    """
    Matrix multiplication of A (M, K) and B (K, N) resulting in C (M, N).
    Optional activation is applied element-wise on C if specified.
    Supported activation: 'leaky_relu'.
    """
    (M, K) = A.shape
    (K2, N) = B.shape
    assert K == K2, 'Incompatible dimensions for matrix multiplication.'
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    use_activation = activation == 'leaky_relu'
    matmul_kernel[grid](A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, USE_ACTIVATION=use_activation, ALPHA=alpha)
    return C
def test_matmul():
    test_results = {}
    A1 = torch.randn((8, 16), device='cuda', dtype=torch.float32)
    B1 = torch.randn((16, 8), device='cuda', dtype=torch.float32)
    C1 = matmul(A1, B1, activation=None)
    test_results['test_case_1'] = C1.cpu().numpy()
    A2 = torch.randn((8, 16), device='cuda', dtype=torch.float32)
    B2 = torch.randn((16, 8), device='cuda', dtype=torch.float32)
    C2 = matmul(A2, B2, activation='leaky_relu', alpha=0.05)
    test_results['test_case_2'] = C2.cpu().numpy()
    A3 = torch.randn((64, 64), device='cuda', dtype=torch.float32)
    B3 = torch.randn((64, 64), device='cuda', dtype=torch.float32)
    C3 = matmul(A3, B3)
    test_results['test_case_3'] = C3.cpu().numpy()
    A4 = torch.randn((64, 32), device='cuda', dtype=torch.float32)
    B4 = torch.randn((32, 128), device='cuda', dtype=torch.float32)
    C4 = matmul(A4, B4, activation='leaky_relu', alpha=0.01)
    test_results['test_case_4'] = C4.cpu().numpy()
    print('Test results keys:', list(test_results.keys()))
##################################################################################################################################################



def test_matmul():
    results = {}

    # Test case 1: Basic matrix multiplication without activation
    a = torch.randn((256, 64), device='cuda', dtype=torch.float16)
    b = torch.randn((64, 256), device='cuda', dtype=torch.float16)
    c = matmul(a, b)
    results["test_case_1"] = c

    # Test case 2: Matrix multiplication with leaky ReLU activation
    c_with_activation = matmul(a, b, activation="leaky_relu")
    results["test_case_2"] = c_with_activation

    # Test case 3: Matrix multiplication with larger dimensions
    a_large = torch.randn((512, 128), device='cuda', dtype=torch.float16)
    b_large = torch.randn((128, 512), device='cuda', dtype=torch.float16)
    c_large = matmul(a_large, b_large)
    results["test_case_3"] = c_large

    # Test case 4: Matrix multiplication with larger dimensions and leaky ReLU activation
    c_large_with_activation = matmul(a_large, b_large, activation="leaky_relu")
    results["test_case_4"] = c_large_with_activation

    return results

result_gold = test_matmul()
