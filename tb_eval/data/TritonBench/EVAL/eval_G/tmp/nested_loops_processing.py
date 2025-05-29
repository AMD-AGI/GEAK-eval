import torch
import triton
import triton.language as tl

@triton.jit
def nested3(in_ptr, out_ptr, stride_m: tl.constexpr, stride_n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_x = tl.program_id(0)
    col_offset = block_x * 4
    col_indices = col_offset + tl.arange(0, 4)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                offset = i * stride_m + j * stride_n + col_indices + k
                vals = tl.load(in_ptr + offset)
                tl.store(out_ptr + offset, vals)
def wrapper_nested3(n_rows, n_cols):
    x = torch.randn((n_rows, n_cols), device='cuda', dtype=torch.float32)
    output = torch.zeros_like(x)
    grid = lambda META: (n_cols // 4,)
    nested3[grid](x, output, stride_m=n_cols, stride_n=1, BLOCK_SIZE=4)
    return output
def test_nested3():
    results = {}
    out1 = wrapper_nested3(4, 8)
    results['test_case_1'] = out1.clone().cpu().numpy()
    out2 = wrapper_nested3(2, 4)
    results['test_case_2'] = out2.clone().cpu().numpy()
    out3 = wrapper_nested3(6, 8)
    results['test_case_3'] = out3.clone().cpu().numpy()
    out4 = wrapper_nested3(8, 8)
    results['test_case_4'] = out4.clone().cpu().numpy()
    print(results)
##################################################################################################################################################



import torch

def test_nested3():
    # Test dimensions
    results = {}
    
    # Test case 1
    n_rows = 8
    n_cols = 8
    results['test_case_1'] = wrapper_nested3(n_rows, n_cols)
    
    # Test case 2
    n_rows = 4
    n_cols = 4
    results['test_case_2'] = wrapper_nested3(n_rows, n_cols)
    
    # Test case 3
    n_rows = 16
    n_cols = 16
    results['test_case_3'] = wrapper_nested3(n_rows, n_cols)
    
    # Test case 4
    n_rows = 2
    n_cols = 2
    results['test_case_4'] = wrapper_nested3(n_rows, n_cols)
    
    return results

result_gold = test_nested3()
