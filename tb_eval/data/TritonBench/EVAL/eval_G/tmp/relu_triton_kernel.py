import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + offsets, mask=offsets < N, other=0.0)
    x_relu = tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, x_relu, mask=offsets < N)
def relu(x: torch.Tensor, block_size: int=1024):
    y = torch.empty_like(x)
    grid = lambda meta: ((x.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    relu_kernel[grid](x, y, x.numel(), BLOCK_SIZE=block_size)
    return y
def test_relu():
    test_results = {}
    inp1 = torch.tensor([-3.0, -2.0, -1.0], device='cuda')
    out1 = relu(inp1)
    test_results['test_case_1'] = out1.cpu().numpy()
    inp2 = torch.tensor([-1.0, 0.0, 1.0, 2.0], device='cuda')
    out2 = relu(inp2)
    test_results['test_case_2'] = out2.cpu().numpy()
    inp3 = torch.randn(1024, device='cuda') * 5 - 2.5
    out3 = relu(inp3)
    test_results['test_case_3'] = out3.cpu().numpy()
    inp4 = torch.tensor([0.0, 10.0, 0.0, 3.5], device='cuda')
    out4 = relu(inp4)
    test_results['test_case_4'] = out4.cpu().numpy()
    print(test_results)
##################################################################################################################################################



import torch

def test_relu():
    results = {}
    
    # Test case 1: All negative values
    input_tensor = torch.tensor([-3.0, -1.0, -0.5, -2.0, -5.0], dtype=torch.float32, device='cuda')
    output_tensor = relu(input_tensor)
    results['test_case_1'] = output_tensor

    # Test case 2: All positive values
    input_tensor = torch.tensor([3.0, 1.0, 0.5, 2.0, 5.0], dtype=torch.float32, device='cuda')
    output_tensor = relu(input_tensor)
    results['test_case_2'] = output_tensor

    # Test case 3: Mixed values
    input_tensor = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0], dtype=torch.float32, device='cuda')
    output_tensor = relu(input_tensor)
    results['test_case_3'] = output_tensor

    # Test case 4: Zero values
    input_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
    output_tensor = relu(input_tensor)
    results['test_case_4'] = output_tensor

    return results

result_gold = test_relu()
