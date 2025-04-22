import torch
import math
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_input = input_ptr + row_idx * input_row_stride
    row_start_output = output_ptr + row_idx * output_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_data = tl.load(row_start_input + offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(row_data, axis=0)
    row_data = row_data - row_max
    row_exp = tl.exp(row_data)
    row_sum = tl.sum(row_exp, axis=0)
    row_softmax = row_exp / row_sum
    tl.store(row_start_output + offsets, row_softmax, mask=mask)
def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute row-wise softmax using Triton.
    """
    assert x.is_cuda, 'Input tensor must be on GPU'
    (rows, cols) = x.shape
    BLOCK_SIZE = 2 ** math.ceil(math.log2(cols)) if cols > 1 else 1
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    num_warps = 4
    if BLOCK_SIZE <= 64:
        num_warps = 1
    elif BLOCK_SIZE <= 128:
        num_warps = 2
    elif BLOCK_SIZE <= 256:
        num_warps = 4
    elif BLOCK_SIZE <= 512:
        num_warps = 8
    else:
        num_warps = 8
    out = torch.empty_like(x)
    grid = (rows,)
    softmax_kernel[grid](x, out, x.stride(0), out.stride(0), cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=2)
    return out
def test_softmax():
    """
    Single test function with up to four branch tests.
    All tests store their results in a dictionary with 'test_case_n' as keys.
    """
    results = {}
    x1 = torch.tensor([[0.1, 0.2, 0.3]], device='cuda')
    out1 = softmax(x1)
    ref1 = torch.nn.functional.softmax(x1, dim=1)
    diff1 = (out1 - ref1).abs().max().item()
    results['test_case_1'] = diff1
    x2 = torch.randn((2, 5), device='cuda')
    out2 = softmax(x2)
    ref2 = torch.nn.functional.softmax(x2, dim=1)
    diff2 = (out2 - ref2).abs().max().item()
    results['test_case_2'] = diff2
    x3 = torch.randn((4, 17), device='cuda')
    out3 = softmax(x3)
    ref3 = torch.nn.functional.softmax(x3, dim=1)
    diff3 = (out3 - ref3).abs().max().item()
    results['test_case_3'] = diff3
    x4 = torch.randn((7, 32), device='cuda')
    out4 = softmax(x4)
    ref4 = torch.nn.functional.softmax(x4, dim=1)
    diff4 = (out4 - ref4).abs().max().item()
    results['test_case_4'] = diff4
    print(results)
##################################################################################################################################################



import torch

def test_softmax():
    # Define the input tensor
    x = torch.randn(128, 512, device='cuda', dtype=torch.float32)

    # Compute softmax using Triton
    output = softmax(x)

    # Additional test cases to cover all branches
    results = {}

    # Test case 1: n_cols < 2048
    x1 = torch.randn(128, 1024, device='cuda', dtype=torch.float32)
    results['test_case_1'] = softmax(x1)

    # Test case 2: 2048 <= n_cols < 4096
    x2 = torch.randn(128, 2048, device='cuda', dtype=torch.float32)
    results['test_case_2'] = softmax(x2)

    # Test case 3: n_cols >= 4096
    x3 = torch.randn(128, 4096, device='cuda', dtype=torch.float32)
    results['test_case_3'] = softmax(x3)

    # Test case 4: n_cols < 2048 (original test case)
    results['test_case_4'] = output

    return results

result_gold = test_softmax()
