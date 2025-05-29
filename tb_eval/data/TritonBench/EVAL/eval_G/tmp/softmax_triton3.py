import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_start + col_offsets
    row_data = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_max = tl.maximum(tl.max(row_data, axis=0), 0.0)
    row_data = row_data - row_max
    if mask_ptr is not None:
        mask_val = tl.load(mask_ptr + row_start + col_offsets, mask=col_offsets < n_cols, other=False)
        row_data = tl.where(mask_val, -float('inf'), row_data)
    row_exp = tl.exp(row_data)
    denom = tl.sum(row_exp, axis=0)
    row_softmax = row_exp / denom
    output_ptrs = output_ptr + row_start + col_offsets
    tl.store(output_ptrs, row_softmax, mask=col_offsets < n_cols)
def softmax(input: torch.Tensor, mask: torch.Tensor=None, dim=-1) -> torch.Tensor:
    """
    Computes the softmax of the input tensor along the specified dimension using Triton.
    """
    assert input.device.type == 'cuda', 'Input must be on CUDA device.'
    if mask is not None:
        assert mask.device.type == 'cuda', 'Mask must be on CUDA device.'
        assert mask.shape == input.shape, 'Mask shape must match input shape.'
    assert dim == -1, 'Only dim = -1 is supported for this Triton softmax.'
    original_shape = input.shape
    num_dims = len(original_shape)
    n_rows = 1
    n_cols = original_shape[-1]
    for d in range(num_dims - 1):
        n_rows *= original_shape[d]
    input_2d = input.view(n_rows, n_cols)
    mask_2d = mask.view(n_rows, n_cols) if mask is not None else None
    output_2d = torch.empty_like(input_2d)
    row_stride = input_2d.stride(0)
    BLOCK_SIZE = 128
    grid = (n_rows,)
    softmax_kernel[grid](output_2d, input_2d, row_stride, n_cols, mask_2d, BLOCK_SIZE=BLOCK_SIZE)
    return output_2d.view(*original_shape)
def test_softmax():
    """
    Test function to cover up to 4 branches for the softmax kernel and wrapper.
    Stores test outputs in a dictionary keyed by 'test_case_n'.
    """
    results = {}
    x1 = torch.randn(2, 3, device='cuda')
    out1 = softmax(x1)
    results['test_case_1'] = out1
    x2 = torch.randn(2, 3, device='cuda')
    mask2 = torch.tensor([[False, True, False], [True, False, True]], dtype=torch.bool, device='cuda')
    out2 = softmax(x2, mask2)
    results['test_case_2'] = out2
    x3 = torch.randn(2, 2, 2, device='cuda')
    out3 = softmax(x3)
    results['test_case_3'] = out3
    x4 = torch.randn(256, 128, device='cuda')
    out4 = softmax(x4)
    results['test_case_4'] = out4
    for (k, v) in results.items():
        print(k, v)
##################################################################################################################################################



def test_softmax():
    # Test Case 1: Small matrix without mask
    input_tensor_1 = torch.randn(32, 128, dtype=torch.float16, device='cuda')
    output_tensor_1 = softmax(input_tensor_1)

    # Test Case 2: Small matrix with mask
    input_tensor_2 = torch.randn(32, 128, dtype=torch.float16, device='cuda')
    mask_tensor_2 = torch.randint(0, 2, (32, 128), dtype=torch.float16, device='cuda')
    output_tensor_2 = softmax(input_tensor_2, mask=mask_tensor_2)

    # Test Case 3: Larger matrix without mask
    input_tensor_3 = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    output_tensor_3 = softmax(input_tensor_3)

    # Test Case 4: Larger matrix with mask
    input_tensor_4 = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    mask_tensor_4 = torch.randint(0, 2, (1024, 512), dtype=torch.float16, device='cuda')
    output_tensor_4 = softmax(input_tensor_4, mask=mask_tensor_4)

    # Test Case 5: Very large matrix without mask
    input_tensor_5 = torch.randn(100000, 256, dtype=torch.float16, device='cuda')
    output_tensor_5 = softmax(input_tensor_5)

    # Test Case 6: Very large matrix with mask
    input_tensor_6 = torch.randn(100000, 256, dtype=torch.float16, device='cuda')
    mask_tensor_6 = torch.randint(0, 2, (100000, 256), dtype=torch.float16, device='cuda')
    output_tensor_6 = softmax(input_tensor_6, mask=mask_tensor_6)

    return {
        "test_case_1": output_tensor_1,
        "test_case_2": output_tensor_2,
        "test_case_3": output_tensor_3,
        "test_case_4": output_tensor_4,
        "test_case_5": output_tensor_5,
        "test_case_6": output_tensor_6
    }

# Run the test function
result_gold = test_softmax()
