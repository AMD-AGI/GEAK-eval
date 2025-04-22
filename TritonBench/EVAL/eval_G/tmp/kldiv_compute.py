import torch
import triton
import triton.language as tl

@triton.jit
def kldivergence_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=1.0)
    eps = 1e-09
    output = x * tl.log((x + eps) / (y + eps))
    tl.store(output_ptr + offsets, output, mask=mask)
def kldivergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError('Input tensors must be on GPU.')
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kldivergence_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
def test_kldivergence():
    results = {}
    x1 = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32, device='cuda')
    y1 = torch.tensor([0.3, 0.4, 0.3], dtype=torch.float32, device='cuda')
    out1 = kldivergence(x1, y1)
    results['test_case_1'] = out1.cpu().tolist()
    x2 = torch.rand(256, dtype=torch.float32, device='cuda')
    y2 = torch.rand(256, dtype=torch.float32, device='cuda') + 1e-08
    out2 = kldivergence(x2, y2)
    results['test_case_2'] = out2.cpu().tolist()
    x3 = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32, device='cuda')
    y3 = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32, device='cuda')
    out3 = kldivergence(x3, y3)
    results['test_case_3'] = out3.cpu().tolist()
    x4 = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32, device='cuda')
    y4 = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32, device='cuda')
    out4 = kldivergence(x4, y4)
    results['test_case_4'] = out4.cpu().tolist()
    print(results)
##################################################################################################################################################



import torch

def test_kldivergence():
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    # 使用 Triton 计算 KL 散度
    output_triton = kldivergence(x, y)

    # 分支覆盖率【1/4】

    # 补全所有分支调用
    results = {}
    
    # Test case 1
    x1 = torch.rand(1024, device='cuda')
    y1 = torch.rand(1024, device='cuda')
    results['test_case_1'] = kldivergence(x1, y1)

    # Test case 2
    x2 = torch.rand(2048, device='cuda')
    y2 = torch.rand(2048, device='cuda')
    results['test_case_2'] = kldivergence(x2, y2)

    # Test case 3
    x3 = torch.rand(4096, device='cuda')
    y3 = torch.rand(4096, device='cuda')
    results['test_case_3'] = kldivergence(x3, y3)

    # Test case 4
    x4 = torch.rand(8192, device='cuda')
    y4 = torch.rand(8192, device='cuda')
    results['test_case_4'] = kldivergence(x4, y4)

    return results

result_gold = test_kldivergence()
