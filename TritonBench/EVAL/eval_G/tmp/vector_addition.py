import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output_vals = x_vals + y_vals
    tl.store(output_ptr + offsets, output_vals, mask=mask)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, 'x must be a CUDA tensor'
    assert y.is_cuda, 'y must be a CUDA tensor'
    n_elements = x.numel()
    assert n_elements == y.numel(), 'Input tensors must have the same number of elements'
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
def test_add():
    test_results = {}
    x1 = torch.tensor([1, 2, 3], device='cuda', dtype=torch.float32)
    y1 = torch.tensor([4, 5, 6], device='cuda', dtype=torch.float32)
    out1 = add(x1, y1)
    test_results['test_case_1'] = out1.cpu().tolist()
    x2 = torch.tensor([], device='cuda', dtype=torch.float32)
    y2 = torch.tensor([], device='cuda', dtype=torch.float32)
    out2 = add(x2, y2)
    test_results['test_case_2'] = out2.cpu().tolist()
    x3 = torch.arange(1024, device='cuda', dtype=torch.float32)
    y3 = torch.ones(1024, device='cuda', dtype=torch.float32)
    out3 = add(x3, y3)
    test_results['test_case_3'] = out3.cpu().tolist()
    x4 = torch.tensor([10], device='cuda', dtype=torch.float32)
    y4 = torch.tensor([-5], device='cuda', dtype=torch.float32)
    out4 = add(x4, y4)
    test_results['test_case_4'] = out4.cpu().tolist()
    for (k, v) in test_results.items():
        print(f'{k} = {v}')
##################################################################################################################################################



def test_add():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    # Test case 1
    output_triton_1 = add(x, y)
    
    # Test case 2
    size_2 = 1024
    x_2 = torch.rand(size_2, device='cuda')
    y_2 = torch.rand(size_2, device='cuda')
    output_triton_2 = add(x_2, y_2)
    
    # Test case 3
    size_3 = 2048
    x_3 = torch.rand(size_3, device='cuda')
    y_3 = torch.rand(size_3, device='cuda')
    output_triton_3 = add(x_3, y_3)
    
    # Test case 4
    size_4 = 4096
    x_4 = torch.rand(size_4, device='cuda')
    y_4 = torch.rand(size_4, device='cuda')
    output_triton_4 = add(x_4, y_4)
    
    results = {
        "test_case_1": output_triton_1,
        "test_case_2": output_triton_2,
        "test_case_3": output_triton_3,
        "test_case_4": output_triton_4
    }
    
    return results

result_gold = test_add()
