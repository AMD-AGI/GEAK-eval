import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    prog_id = tl.program_id(0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a_vals = tl.load(A + offs, mask=mask, other=0.0)
    b_vals = tl.load(B + offs, mask=mask, other=0.0)
    c_vals = a_vals + b_vals
    tl.store(C + offs, c_vals, mask=mask)
def custom_add(a: torch.Tensor, b: torch.Tensor):
    c = torch.empty_like(a)
    size = a.numel()
    BLOCK = 16
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK']),)
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    return c
def test_add():
    test_results = {}
    a1 = torch.randn(16, device='cuda')
    b1 = torch.randn(16, device='cuda')
    c1 = custom_add(a1, b1)
    test_results['test_case_1'] = c1.cpu()
    a2 = torch.randn(100, device='cuda')
    b2 = torch.randn(100, device='cuda')
    c2 = custom_add(a2, b2)
    test_results['test_case_2'] = c2.cpu()
    a3 = torch.randn((8, 8), device='cuda')
    b3 = torch.randn((8, 8), device='cuda')
    c3 = custom_add(a3, b3)
    test_results['test_case_3'] = c3.cpu()
    a4 = torch.randn(103, device='cuda')
    b4 = torch.randn(103, device='cuda')
    c4 = custom_add(a4, b4)
    test_results['test_case_4'] = c4.cpu()
    print(test_results)
##################################################################################################################################################



import torch

def test_add():
    # 测试用例 1：简单的两个向量加法
    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=torch.float32, device='cuda')
    b = torch.tensor([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=torch.float32, device='cuda')
    c = custom_add(a, b)

    # 测试用例 2：不同值的加法
    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32, device='cuda')
    b = torch.tensor([8, 7, 6, 5, 4, 3, 2, 1], dtype=torch.float32, device='cuda')
    c = custom_add(a, b)

    # 测试用例 3：更大向量的加法
    a = torch.arange(32, dtype=torch.float32, device='cuda')
    b = torch.arange(32, 0, -1, dtype=torch.float32, device='cuda')
    c = custom_add(a, b)

    # 测试用例 4：空向量的边界情况
    a = torch.tensor([], dtype=torch.float32, device='cuda')
    b = torch.tensor([], dtype=torch.float32, device='cuda')
    c = custom_add(a, b)

    test_results = {
        "test_case_1": custom_add(a, b),
        "test_case_2": custom_add(a, b),
        "test_case_3": custom_add(a, b),
        "test_case_4": custom_add(a, b),
    }
    return test_results

result_gold = test_add()
