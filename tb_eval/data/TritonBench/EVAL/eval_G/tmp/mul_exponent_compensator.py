import torch
import math
import triton
import triton.language as tl

@triton.jit
def mul_kernel(src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = pid * BLOCK_SIZE + offsets
    src_vals = tl.load(src_ptr + indices, mask=indices < BLOCK_SIZE)
    compensator = 2.0 ** (127 - 15)
    result = src_vals * compensator
    tl.store(dst_ptr + indices, result, mask=indices < BLOCK_SIZE)
def launch_mul_kernel(src: torch.Tensor, BLOCK_SIZE: int=256) -> torch.Tensor:
    """
    Launches the mul_kernel to multiply each element of src
    with the exponent compensator 2.0^(127-15).

    Args:
        src (torch.Tensor): Input tensor (on GPU).
        BLOCK_SIZE (int, optional): Block size. Defaults to 256.

    Returns:
        torch.Tensor: Output tensor (on GPU) after multiplication.
    """
    assert src.is_cuda, 'Input tensor must be on CUDA.'
    dst = torch.empty_like(src, device=src.device)
    grid = (src.shape[0] // BLOCK_SIZE,)
    mul_kernel[grid](src, dst, BLOCK_SIZE=BLOCK_SIZE)
    return dst
def test_mul_kernel():
    """
    Test function for mul_kernel and launch_mul_kernel. 
    Executes multiple branches (up to 4), stores 
    results in a dictionary, and returns them.
    """
    results = {}
    src1 = torch.randn(256, device='cuda', dtype=torch.float32)
    out1 = launch_mul_kernel(src1)
    results['test_case_1'] = out1.clone()
    src2 = torch.randn(512, device='cuda', dtype=torch.float32)
    out2 = launch_mul_kernel(src2, BLOCK_SIZE=128)
    results['test_case_2'] = out2.clone()
    src3 = torch.randn(1024, device='cuda', dtype=torch.float32)
    out3 = launch_mul_kernel(src3)
    results['test_case_3'] = out3.clone()
    src4 = torch.randn(128, device='cuda', dtype=torch.float32)
    out4 = launch_mul_kernel(src4, BLOCK_SIZE=128)
    results['test_case_4'] = out4.clone()
    print('Test results:', results)
##################################################################################################################################################



def test_mul():
    src = torch.tensor([8323072], dtype=torch.int32, device='cuda').view(torch.float32)
    
    test_cases = {}
    
    # Test case 1
    dst_triton_1 = launch_mul_kernel(src, BLOCK_SIZE=1)
    test_cases['test_case_1'] = dst_triton_1

    # Test case 2
    dst_triton_2 = launch_mul_kernel(src, BLOCK_SIZE=2)
    test_cases['test_case_2'] = dst_triton_2
    
    # Test case 3
    dst_triton_3 = launch_mul_kernel(src, BLOCK_SIZE=4)
    test_cases['test_case_3'] = dst_triton_3
    
    # Test case 4
    dst_triton_4 = launch_mul_kernel(src, BLOCK_SIZE=8)
    test_cases['test_case_4'] = dst_triton_4

    return test_cases

result_gold = test_mul()
