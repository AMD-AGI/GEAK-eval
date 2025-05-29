import torch
import triton
import triton.language as tl
from torch.testing import assert_close

@triton.jit
def load_reduce_kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_y, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    m_range = tl.arange(0, BLOCK_M)
    n_range = tl.arange(0, BLOCK_N)
    x_offset = pid_m * BLOCK_M + m_range
    row_offsets = x_offset[:, None] * stride_xm
    col_offsets = n_range[None, :] * stride_xn
    x_block_ptr = x_ptr + row_offsets + col_offsets
    block = tl.load(x_block_ptr, mask=(x_offset[:, None] < stride_y) & (n_range[None, :] < BLOCK_N), other=-float('inf'))
    row_max = tl.max(block, axis=1)
    y_offset = pid_m * BLOCK_M + m_range
    y_ptr_out = y_ptr + y_offset * stride_y
    tl.store(y_ptr_out, row_max, mask=y_offset < stride_y)
def load_reduce(x: torch.Tensor):
    """
    Wrapper that calls `load_reduce_kernel` on x, to compute row-wise maximum.
    """
    (M, N) = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    BLOCK_M = 32
    BLOCK_N = 32
    grid = ((M + BLOCK_M - 1) // BLOCK_M,)
    load_reduce_kernel[grid](x, y, x.stride(0), x.stride(1), y.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return y
def test_load_reduce():
    """
    A single test function with up to 4 branch tests.
    """
    results = {}
    x1 = torch.randn((32, 32), device='cuda')
    y1 = load_reduce(x1)
    (expected1, _) = x1.max(dim=1)
    assert_close(y1, expected1)
    results['test_case_1'] = 'Success'
    x2 = torch.randn((50, 27), device='cuda')
    y2 = load_reduce(x2)
    (expected2, _) = x2.max(dim=1)
    assert_close(y2, expected2)
    results['test_case_2'] = 'Success'
    x3 = torch.randn((128, 64), device='cuda')
    y3 = load_reduce(x3)
    (expected3, _) = x3.max(dim=1)
    assert_close(y3, expected3)
    results['test_case_3'] = 'Success'
    x4 = torch.randn((16, 100), device='cuda')
    y4 = load_reduce(x4)
    (expected4, _) = x4.max(dim=1)
    assert_close(y4, expected4)
    results['test_case_4'] = 'Success'
    print(results)
##################################################################################################################################################



import torch

def test_reduce():
    # 测试参数设置
    test_cases = [
        {"BLOCK_M": 16, "BLOCK_N": 16, "dtype_str": "float16"},
        {"BLOCK_M": 32, "BLOCK_N": 32, "dtype_str": "float16"},
        {"BLOCK_M": 64, "BLOCK_N": 64, "dtype_str": "float32"},
        {"BLOCK_M": 128, "BLOCK_N": 128, "dtype_str": "float32"},
    ]

    results = {}
    for i, case in enumerate(test_cases):
        BLOCK_M = case["BLOCK_M"]
        BLOCK_N = case["BLOCK_N"]
        dtype_str = case["dtype_str"]

        try:
            load_reduce(BLOCK_M, BLOCK_N, dtype_str)
            results[f"test_case_{i+1}"] = "passed"
        except Exception as e:
            results[f"test_case_{i+1}"] = f"failed: {e}"

    return results

result_gold = test_reduce()
