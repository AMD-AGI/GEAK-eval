import torch
import triton
import triton.language as tl

@triton.jit
def puzzle1_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr, value: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x_data = tl.load(x_ptr + offsets, mask=mask)
    y_data = x_data + value
    tl.store(output_ptr + offsets, y_data, mask=mask)
def puzzle1(x: torch.Tensor):
    if not x.is_cuda:
        raise ValueError('Input tensor must be on GPU (CUDA).')
    output = torch.empty_like(x)
    if not output.is_cuda:
        raise ValueError('Output tensor must be on the same GPU (CUDA).')
    N = x.numel()

    def grid(META):
        return ((N + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'],)
    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=128, value=10)
    return output
def test_puzzle1():
    test_results = {}
    x1 = torch.tensor([1, 2, 3, 4], device='cuda')
    out1 = puzzle1(x1)
    test_results['test_case_1'] = out1.tolist()
    x2 = torch.zeros(8, device='cuda')
    out2 = puzzle1(x2)
    test_results['test_case_2'] = out2.tolist()
    x3 = torch.arange(8, dtype=torch.float32, device='cuda')
    out3 = puzzle1(x3)
    test_results['test_case_3'] = out3.tolist()
    x4 = torch.tensor([], device='cuda')
    out4 = puzzle1(x4)
    test_results['test_case_4'] = out4.tolist()
    print(test_results)
def grid(META):
    return ((N + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'],)
##################################################################################################################################################



import torch

def test_puzzle():
    results = {}
    
    # Test case 1
    a1 = torch.Tensor([4, 5, 3, 2]).to(device=torch.device('cuda'))
    triton_output1 = puzzle1(a1)
    results['test_case_1'] = triton_output1
    
    # Test case 2
    a2 = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8]).to(device=torch.device('cuda'))
    triton_output2 = puzzle1(a2)
    results['test_case_2'] = triton_output2
    
    # Test case 3
    a3 = torch.Tensor([10, 20, 30]).to(device=torch.device('cuda'))
    triton_output3 = puzzle1(a3)
    results['test_case_3'] = triton_output3
    
    # Test case 4
    a4 = torch.Tensor([0, -1, -2, -3]).to(device=torch.device('cuda'))
    triton_output4 = puzzle1(a4)
    results['test_case_4'] = triton_output4
    
    return results

result_gold = test_puzzle()
