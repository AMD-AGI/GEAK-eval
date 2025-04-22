import torch
import triton
import triton.language as tl

@triton.jit
def masked_add_kernel(grad_ptr, p_ptr, p_mask_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < n_elements
    p_mask_vals = tl.load(p_mask_ptr + block_start, mask=mask)
    bool_mask = p_mask_vals.to(tl.int1)
    grad_vals = tl.load(grad_ptr + block_start, mask=mask)
    p_vals = tl.load(p_ptr + block_start, mask=mask)
    out = grad_vals + bool_mask * (p_vals * alpha)
    tl.store(grad_ptr + block_start, out, mask=mask)
def masked_add(grad, p, p_mask, alpha=1.0):
    assert grad.is_cuda and p.is_cuda and p_mask.is_cuda, 'All tensors must be on CUDA'
    assert grad.is_contiguous() and p.is_contiguous() and p_mask.is_contiguous(), 'All tensors must be contiguous'
    n_elements = grad.numel()
    assert p.numel() == n_elements and p_mask.numel() == n_elements, 'All tensors must have the same number of elements'
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    masked_add_kernel[grid](grad, p, p_mask, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE)
def test_masked_add():
    """
    Single test function with up to 4 branch tests.
    Stores outputs in a dictionary under keys 'test_case_n'.
    """
    results = {}
    grad1 = torch.randn(10, device='cuda')
    p1 = torch.randn(10, device='cuda')
    p_mask1 = torch.randint(0, 2, (10,), device='cuda', dtype=torch.int32)
    masked_add(grad1, p1, p_mask1)
    results['test_case_1'] = grad1.clone().cpu()
    grad2 = torch.randn(10, device='cuda')
    p2 = torch.randn(10, device='cuda')
    p_mask2 = torch.randint(0, 2, (10,), device='cuda', dtype=torch.int32)
    masked_add(grad2, p2, p_mask2, alpha=2.0)
    results['test_case_2'] = grad2.clone().cpu()
    grad3 = torch.randn(2000, device='cuda')
    p3 = torch.randn(2000, device='cuda')
    p_mask3 = torch.randint(0, 2, (2000,), device='cuda', dtype=torch.int32)
    masked_add(grad3, p3, p_mask3)
    results['test_case_3'] = grad3[:10].clone().cpu()
    grad4 = torch.randn(10, device='cuda')
    p4 = torch.randn(10, device='cuda')
    p_mask4 = torch.zeros(10, device='cuda', dtype=torch.int32)
    masked_add(grad4, p4, p_mask4)
    results['test_case_4'] = grad4.clone().cpu()
    print('All test case results:')
    for (k, v) in results.items():
        print(k, v)
##################################################################################################################################################



import torch

# 测试代码
def test_masked_add():
    # 设置随机种子以保证结果可复现
    torch.manual_seed(0)
    n = 10000  # 选择较大的张量大小

    # 生成随机张量
    grad = torch.randn(n, device='cuda')
    p_data = torch.randn(n, device='cuda')
    p_mask = torch.randint(0, 2, (n,), device='cuda')  # 生成0或1的掩码

    # Triton版本
    results = {}
    
    # Test case 1
    grad_triton = grad.clone()
    masked_add(grad_triton, p_data, p_mask, alpha=0.5)
    results['test_case_1'] = grad_triton.clone()

    # Test case 2: alpha = 0
    grad_triton = grad.clone()
    masked_add(grad_triton, p_data, p_mask, alpha=0)
    results['test_case_2'] = grad_triton.clone()

    # Test case 3: all mask values are 0
    p_mask_zero = torch.zeros(n, device='cuda', dtype=torch.int32)
    grad_triton = grad.clone()
    masked_add(grad_triton, p_data, p_mask_zero, alpha=0.5)
    results['test_case_3'] = grad_triton.clone()

    # Test case 4: all mask values are 1
    p_mask_one = torch.ones(n, device='cuda', dtype=torch.int32)
    grad_triton = grad.clone()
    masked_add(grad_triton, p_data, p_mask_one, alpha=0.5)
    results['test_case_4'] = grad_triton.clone()

    return results

# 运行测试
result_gold = test_masked_add()
