import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(axis=0)
    row_input_offset = row_id * input_row_stride
    row_output_offset = row_id * output_row_stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    data = tl.load(input_ptr + row_input_offset + cols, mask=mask, other=-float('inf'))
    max_val = tl.max(data, axis=0)
    data = data - max_val
    numerator = tl.exp(data)
    denominator = tl.sum(numerator, axis=0)
    softmax_result = numerator / denominator
    tl.store(output_ptr + row_output_offset + cols, softmax_result, mask=mask)
def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-based softmax wrapper function.
    """
    assert x.is_cuda, 'Input tensor must be on GPU.'
    (n_rows, n_cols) = x.shape
    output = torch.empty_like(x)
    block_size = 1
    while block_size < n_cols and block_size < 1024:
        block_size *= 2
    grid = (n_rows,)
    softmax_kernel[grid](x, output, x.stride(0), output.stride(0), n_cols, BLOCK_SIZE=block_size)
    return output
def test_softmax():
    """
    Single function containing multiple branch tests. No parameters.
    """
    results = {}
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    out1 = triton_softmax(x1)
    ref1 = torch.softmax(x1, dim=1)
    results['test_case_1'] = (out1 - ref1).abs().max().item()
    torch.manual_seed(0)
    x2 = torch.randn((128, 64), device='cuda')
    out2 = triton_softmax(x2)
    ref2 = torch.softmax(x2, dim=1)
    results['test_case_2'] = (out2 - ref2).abs().max().item()
    x3 = torch.randn((5, 5), device='cuda')
    out3 = triton_softmax(x3)
    ref3 = torch.softmax(x3, dim=1)
    results['test_case_3'] = (out3 - ref3).abs().max().item()
    x4 = torch.randn((3, 1025), device='cuda')
    out4 = triton_softmax(x4)
    ref4 = torch.softmax(x4, dim=1)
    results['test_case_4'] = (out4 - ref4).abs().max().item()
    print(results)
##################################################################################################################################################



# Test cases for the triton_softmax function
def test_triton_softmax():
    results = {}
    
    # Test case 1: Simple 2x2 matrix
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda")
    output1 = triton_softmax(x1)
    results['test_case_1'] = output1

    # Test case 2: 3x3 matrix with negative values
    x2 = torch.tensor([[-1.0, -2.0, -3.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=torch.float32, device="cuda")
    output2 = triton_softmax(x2)
    results['test_case_2'] = output2

    # Test case 3: 4x4 matrix with larger values
    x3 = torch.tensor([[10.0, 20.0, 30.0, 40.0], [5.0, 15.0, 25.0, 35.0], [0.0, 0.0, 0.0, 0.0], [-10.0, -20.0, -30.0, -40.0]], dtype=torch.float32, device="cuda")
    output3 = triton_softmax(x3)
    results['test_case_3'] = output3

    # Test case 4: 1x5 matrix (single row)
    x4 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32, device="cuda")
    output4 = triton_softmax(x4)
    results['test_case_4'] = output4

    # Test case 5: 5x1 matrix (single column)
    x5 = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32, device="cuda")
    output5 = triton_softmax(x5)
    results['test_case_5'] = output5

    return results

result_gold = test_triton_softmax()
