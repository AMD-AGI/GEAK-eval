import torch
import triton
import triton.language as tl
import math

@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    r = tl.rand(seed, offsets)
    keep = r > p
    x_kept = x * keep * (1.0 / (1 - p))
    tl.store(output_ptr + offsets, x_kept, mask=mask)
def seeded_dropout(x, p, seed=0, block_size=1024):
    """
    Apply seeded dropout to the tensor x with probability p, random seed, and block_size.
    """
    output = torch.empty_like(x)
    x_c = x.contiguous()
    n_elements = x.numel()
    grid = lambda meta: (math.ceil(n_elements / meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x_ptr=x_c, output_ptr=output, n_elements=n_elements, p=p, seed=seed, BLOCK_SIZE=block_size)
    return output
def test_seeded_dropout():
    """
    Test the seeded_dropout function with different branches in a single test function.
    """
    results = {}
    x1 = torch.randn(8, device='cuda')
    out1 = seeded_dropout(x1, p=0.0, seed=42)
    results['test_case_1'] = out1.cpu()
    x2 = torch.randn(4, 4, device='cuda')
    out2 = seeded_dropout(x2, p=0.5)
    results['test_case_2'] = out2.cpu()
    x3 = torch.randn(1024, device='cuda')
    out3 = seeded_dropout(x3, p=0.9, seed=123)
    results['test_case_3'] = out3.cpu()
    x4 = torch.randn(8, device='cuda')
    out4 = seeded_dropout(x4, p=1.0, seed=999)
    results['test_case_4'] = out4.cpu()
    print(results)
##################################################################################################################################################



import torch

# Test for the seeded_dropout function
def test_seeded_dropout():
    # Input tensor
    x = torch.randn(size=(10,)).cuda()
    results = {}
    # Test with the same seed
    results['test_case_1'] = seeded_dropout(x, p=0.5, seed=123)
    results['test_case_2'] = seeded_dropout(x, p=0.5, seed=123)
    # Test with a different seed
    results['test_case_3'] = seeded_dropout(x, p=0.5, seed=512)
    # Test with a different probability
    results['test_case_4'] = seeded_dropout(x, p=0.3, seed=123)
    return results

# Run tests
result_gold = test_seeded_dropout()
