import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, ACTIVATION: tl.constexpr, ALPHA: tl.constexpr):
    """
    Compute the matrix multiplication C = A x B
    A is of shape (M, K)
    B is of shape (K, N)
    C is of shape (M, N)

    Blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) are processed in parallel.
    Optionally apply leaky ReLU activation with slope ALPHA if ACTIVATION == "leaky_relu".
    Apply ReLU activation if ACTIVATION == "relu".
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    num_k_blocks = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    for k_block_id in range(num_k_blocks):
        k_start = k_block_id * BLOCK_SIZE_K
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (offs_k[None, :] + k_start) * stride_ak)
        b_ptrs = B_ptr + ((offs_k[:, None] + k_start) * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k_start < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k_start < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
    if ACTIVATION == 'leaky_relu':
        zero = tl.zeros_like(accumulator)
        accumulator = tl.where(accumulator > 0, accumulator, accumulator * ALPHA)
    elif ACTIVATION == 'relu':
        accumulator = tl.maximum(accumulator, 0.0)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)
def matmul(A, B, activation=None, alpha=0.1):
    """
    A shape: (M, K)
    B shape: (K, N)
    activation can be: None, 'relu', 'leaky_relu'
    alpha is slope for leaky_relu
    """
    assert A.is_cuda and B.is_cuda, 'A and B must be GPU tensors'
    assert A.dtype in (torch.float16, torch.float32), 'A must be float16 or float32'
    assert B.dtype in (torch.float16, torch.float32), 'B must be float16 or float32'
    (M, K) = A.shape
    (Kb, N) = B.shape
    assert K == Kb, 'Inner dimensions must match'
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    ACT = 'none'
    if activation == 'leaky_relu':
        ACT = 'leaky_relu'
    elif activation == 'relu':
        ACT = 'relu'
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, ACTIVATION=ACT, ALPHA=alpha, num_warps=4)
    return C
def test_matmul():
    """
    Test multiple branches of the matmul function in one place.
    Store results in a dictionary test_results for each branch.
    """
    test_results = {}
    A1 = torch.randn(32, 16, device='cuda', dtype=torch.float16)
    B1 = torch.randn(16, 32, device='cuda', dtype=torch.float16)
    C1 = matmul(A1, B1, activation=None)
    test_results['test_case_1'] = C1.cpu()
    A2 = torch.randn(64, 64, device='cuda', dtype=torch.float16)
    B2 = torch.randn(64, 64, device='cuda', dtype=torch.float16)
    C2 = matmul(A2, B2, activation='relu')
    test_results['test_case_2'] = C2.cpu()
    A3 = torch.randn(128, 32, device='cuda', dtype=torch.float32)
    B3 = torch.randn(32, 128, device='cuda', dtype=torch.float32)
    C3 = matmul(A3, B3, activation='leaky_relu', alpha=0.05)
    test_results['test_case_3'] = C3.cpu()
    A4 = torch.randn(60, 80, device='cuda', dtype=torch.float32)
    B4 = torch.randn(80, 50, device='cuda', dtype=torch.float32)
    C4 = matmul(A4, B4, activation=None)
    test_results['test_case_4'] = C4.cpu()
    for (k, v) in test_results.items():
        print(k, v.shape)
##################################################################################################################################################



def test_matmul():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define matrix dimensions
    M, K, N = 64, 128, 64

    # Create random matrices A and B
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    # Compute matrix multiplication using Triton with leaky_relu activation
    c_triton_leaky_relu = matmul(a, b, activation="leaky_relu")

    # Compute matrix multiplication using Triton without activation
    c_triton_no_activation = matmul(a, b, activation="")

    # Store results in a dictionary
    results = {
        "test_case_1": c_triton_leaky_relu,
        "test_case_2": c_triton_no_activation
    }
    
    return results

# Run the test
result_gold = test_matmul()
