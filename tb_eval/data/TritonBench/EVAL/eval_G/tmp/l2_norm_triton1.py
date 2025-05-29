import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(X_ptr, Y_ptr, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    x_offset = pid * stride_x_row
    y_offset = pid * stride_x_row
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    x_data = tl.load(X_ptr + x_offset + offsets, mask=mask, other=0.0)
    squares = x_data * x_data
    sum_squares = tl.sum(squares, axis=0)
    rstd = 1.0 / tl.sqrt(sum_squares + eps)
    y_data = x_data * rstd
    tl.store(Y_ptr + y_offset + offsets, y_data, mask=mask)
def _l2_norm_fwd(x: torch.Tensor, eps: float=1e-05):
    """
    Forward pass of L2 normalization on 2D input tensor x using a single pass.
    """
    if x.dim() != 2:
        raise ValueError('Input must be 2D')
    if not x.is_cuda:
        raise ValueError('Input tensor must be on CUDA device')
    x = x.contiguous()
    (M, N) = x.shape
    element_size = x.element_size()
    max_bytes_per_block = 65536
    BLOCK_N = max_bytes_per_block // element_size
    if N > BLOCK_N:
        raise RuntimeError('N is larger than the max block size. Not supported in this example.')
    y = torch.empty_like(x)
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](x, y, x.stride(0), N, eps, BLOCK_N=BLOCK_N)
    return y
def test_l2_norm_fwd():
    """
    Test the L2 norm kernel with various branch tests.
    All branches are stored in a results dictionary.
    """
    results = {}
    try:
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
        y1 = _l2_norm_fwd(x1)
        results['test_case_1'] = y1.cpu().numpy()
    except Exception as e:
        results['test_case_1'] = f'Error: {e}'
    try:
        arr_size = 128
        x2 = torch.randn((2, arr_size), device='cuda', dtype=torch.float32)
        y2 = _l2_norm_fwd(x2)
        results['test_case_2'] = y2.detach().cpu().numpy()
    except Exception as e:
        results['test_case_2'] = f'Error: {e}'
    try:
        x3 = torch.randn((1, 100000), device='cuda', dtype=torch.float32)
        y3 = _l2_norm_fwd(x3)
        results['test_case_3'] = y3.detach().cpu().numpy()
    except Exception as e:
        results['test_case_3'] = f'Error: {e}'
    try:
        x4 = torch.randn((5, 16), device='cuda', dtype=torch.float32)
        y4 = _l2_norm_fwd(x4, eps=0.0001)
        results['test_case_4'] = y4.cpu().numpy()
    except Exception as e:
        results['test_case_4'] = f'Error: {e}'
    print(results)
##################################################################################################################################################



import torch

# Test the forward L2 normalization
def test_l2_norm_fwd():
    results = {}
    
    # Test case 1
    x1 = torch.randn(4, 8, device='cuda', dtype=torch.float32)
    y1 = _l2_norm_fwd(x1)
    results['test_case_1'] = y1

    # Test case 2: Different batch size
    x2 = torch.randn(2, 8, device='cuda', dtype=torch.float32)
    y2 = _l2_norm_fwd(x2)
    results['test_case_2'] = y2

    # Test case 3: Different feature size
    x3 = torch.randn(4, 4, device='cuda', dtype=torch.float32)
    y3 = _l2_norm_fwd(x3)
    results['test_case_3'] = y3

    # Test case 4: Larger tensor
    x4 = torch.randn(8, 8, device='cuda', dtype=torch.float32)
    y4 = _l2_norm_fwd(x4)
    results['test_case_4'] = y4

    return results

result_gold = test_l2_norm_fwd()
