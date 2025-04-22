import torch
import triton
import triton.language as tl
import math

@triton.jit
def cos_func(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_offset = pid * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_value = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_value = tl.cos(a_value)
    tl.store(b_ptr + offsets, b_value, mask=mask)
def cos(a: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda:
        raise ValueError("Input tensor 'a' must be on GPU.")
    n_elements = a.numel()
    b = torch.empty_like(a)
    block_size_approx = 2 ** math.ceil(math.log2(math.sqrt(n_elements)))
    BLOCK_SIZE = max(1, block_size_approx)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    cos_func[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return b
def test_cos():
    test_results = {}
    a1 = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2], device='cuda')
    b1 = cos(a1)
    test_results['test_case_1'] = b1.tolist()
    a2 = torch.randn(1024, device='cuda')
    b2 = cos(a2)
    test_results['test_case_2'] = b2[:10].tolist()
    a3 = torch.tensor([math.pi / 4], device='cuda')
    b3 = cos(a3)
    test_results['test_case_3'] = b3.tolist()
    a4 = torch.empty(0, device='cuda')
    b4 = cos(a4)
    test_results['test_case_4'] = b4.tolist()
    print(test_results)
##################################################################################################################################################



def test_cos_function():
    # Create test cases with various input sizes
    test_cases = {
        'test_case_1': torch.rand(1024, device='cuda') * 2 * math.pi,
        'test_case_2': torch.rand(2048, device='cuda') * 2 * math.pi,
        'test_case_3': torch.rand(4096, device='cuda') * 2 * math.pi,
        'test_case_4': torch.rand(8192, device='cuda') * 2 * math.pi
    }
    
    results = {}
    
    for case_name, input_tensor in test_cases.items():
        # Compute cosine using Triton
        B_triton = cos(input_tensor)
        results[case_name] = B_triton
    
    return results

# Run the test
result_gold = test_cos_function()
