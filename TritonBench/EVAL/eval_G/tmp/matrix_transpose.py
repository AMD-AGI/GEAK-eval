import torch
import triton
import triton.language as tl

@triton.jit
def kernel(M_ptr, Out_ptr, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    col_start = 0
    rows = row_start + tl.arange(0, BLOCK_SIZE)
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    rr = tl.broadcast_to(rows[:, None], [BLOCK_SIZE, BLOCK_SIZE])
    cc = tl.broadcast_to(cols[None, :], [BLOCK_SIZE, BLOCK_SIZE])
    row_mask = rr < SIZE_M
    col_mask = cc < D_HEAD
    M_offsets = rr * matrix_stridex + cc * matrix_stridey
    Out_offsets = cc * out_stridex + rr * out_stridey
    matrix_vals = tl.where(row_mask & col_mask, tl.load(M_ptr + M_offsets, mask=row_mask & col_mask, other=0.0), 0.0)
    tl.store(Out_ptr + Out_offsets, matrix_vals, mask=row_mask & col_mask)
def wrapper(SIZE_M, D_HEAD):
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')
    BLOCK_SIZE = 16
    grid = lambda META: ((SIZE_M + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'],)
    kernel[grid](matrix, out, matrix.stride(0), matrix.stride(1), out.stride(0), out.stride(1), SIZE_M, D_HEAD, BLOCK_SIZE=BLOCK_SIZE)
    return out
def test_transpose():
    results = {}
    (SIZE_M1, D_HEAD1) = (4, 4)
    out1 = wrapper(SIZE_M1, D_HEAD1)
    expected1 = torch.randn((SIZE_M1, D_HEAD1), dtype=torch.float16, device='cuda')
    expected1.copy_(torch.randn_like(expected1))
    results['test_case1'] = out1.shape == (D_HEAD1, SIZE_M1)
    (SIZE_M2, D_HEAD2) = (8, 2)
    out2 = wrapper(SIZE_M2, D_HEAD2)
    results['test_case2'] = out2.shape == (D_HEAD2, SIZE_M2)
    (SIZE_M3, D_HEAD3) = (32, 16)
    mat3 = torch.randn((SIZE_M3, D_HEAD3), dtype=torch.float16, device='cuda')
    out3 = wrapper(SIZE_M3, D_HEAD3)
    results['test_case3'] = bool(torch.allclose(out3, mat3.T, atol=0.01, rtol=0.01))
    (SIZE_M4, D_HEAD4) = (10, 5)
    mat4 = torch.randn((SIZE_M4, D_HEAD4), dtype=torch.float16, device='cuda')
    out4 = wrapper(SIZE_M4, D_HEAD4)
    results['test_case4'] = bool(torch.allclose(out4, mat4.T, atol=0.01, rtol=0.01))
    print('Test results:', results)
##################################################################################################################################################



import torch

def test_triton_vs_torch():
    results = {}

    # 测试用例 1: 基本矩阵转置 (小矩阵)
    size_m, d_head = 16, 16
    out = wrapper(size_m, d_head)
    results["test_case_1"] = out.clone()

    # 测试用例 2: 非方形矩阵
    size_m, d_head = 32, 64
    out = wrapper(size_m, d_head)
    results["test_case_2"] = out.clone()

    return results


# 运行测试
result_gold = test_triton_vs_torch()
# print(result_gold)