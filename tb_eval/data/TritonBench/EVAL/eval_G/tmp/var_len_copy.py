import triton
import triton.language as tl
import torch

@triton.jit
def var_len_copy_kernel_triton(old_a_start_ptr, old_a_len_ptr, old_a_location_ptr, new_a_start_ptr, new_a_location_ptr, BLOCK_SIZE: tl.constexpr):
    a_id = tl.program_id(0)
    old_start = tl.load(old_a_start_ptr + a_id)
    length = tl.load(old_a_len_ptr + a_id)
    new_start = tl.load(new_a_start_ptr + a_id)
    for i in range(16):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < length
        old_offset = old_start + offset
        new_offset = new_start + offset
        val = tl.load(old_a_location_ptr + old_offset, mask=mask, other=0)
        tl.store(new_a_location_ptr + new_offset, val, mask=mask)
def launch_var_len_copy_triton(old_a_start: torch.Tensor, old_a_len: torch.Tensor, old_a_location: torch.Tensor, new_a_start: torch.Tensor, new_a_location: torch.Tensor, BLOCK_SIZE: int=256):
    """
    Launch the var_len_copy_kernel_triton 
    Grid size = number of segments to copy
    """
    num_segments = old_a_start.shape[0]
    grid = (num_segments,)
    var_len_copy_kernel_triton[grid](old_a_start, old_a_len, old_a_location, new_a_start, new_a_location, BLOCK_SIZE=BLOCK_SIZE)
def test_var_len_copy_kernel():
    results = {}
    test_case_1_key = 'test_case_1'
    BLOCK_SIZE = 256
    old_a_start_1 = torch.tensor([0], dtype=torch.int32, device='cuda')
    old_a_len_1 = torch.tensor([8], dtype=torch.int32, device='cuda')
    new_a_start_1 = torch.tensor([0], dtype=torch.int32, device='cuda')
    old_a_loc_1 = torch.arange(8, dtype=torch.float32, device='cuda')
    new_a_loc_1 = torch.zeros(8, dtype=torch.float32, device='cuda')
    launch_var_len_copy_triton(old_a_start_1, old_a_len_1, old_a_loc_1, new_a_start_1, new_a_loc_1, BLOCK_SIZE)
    results[test_case_1_key] = new_a_loc_1.tolist()
    test_case_2_key = 'test_case_2'
    old_a_start_2 = torch.tensor([2], dtype=torch.int32, device='cuda')
    old_a_len_2 = torch.tensor([0], dtype=torch.int32, device='cuda')
    new_a_start_2 = torch.tensor([1], dtype=torch.int32, device='cuda')
    old_a_loc_2 = torch.arange(10, dtype=torch.float32, device='cuda')
    new_a_loc_2 = torch.zeros(10, dtype=torch.float32, device='cuda')
    launch_var_len_copy_triton(old_a_start_2, old_a_len_2, old_a_loc_2, new_a_start_2, new_a_loc_2, BLOCK_SIZE)
    results[test_case_2_key] = new_a_loc_2.tolist()
    test_case_3_key = 'test_case_3'
    old_a_start_3 = torch.tensor([0, 5], dtype=torch.int32, device='cuda')
    old_a_len_3 = torch.tensor([5, 5], dtype=torch.int32, device='cuda')
    new_a_start_3 = torch.tensor([0, 10], dtype=torch.int32, device='cuda')
    old_a_loc_3 = torch.arange(15, dtype=torch.float32, device='cuda')
    new_a_loc_3 = torch.zeros(20, dtype=torch.float32, device='cuda')
    launch_var_len_copy_triton(old_a_start_3, old_a_len_3, old_a_loc_3, new_a_start_3, new_a_loc_3, BLOCK_SIZE)
    results[test_case_3_key] = new_a_loc_3.tolist()
    test_case_4_key = 'test_case_4'
    old_a_start_4 = torch.tensor([0], dtype=torch.int32, device='cuda')
    old_a_len_4 = torch.tensor([300], dtype=torch.int32, device='cuda')
    new_a_start_4 = torch.tensor([0], dtype=torch.int32, device='cuda')
    old_a_loc_4 = torch.arange(300, dtype=torch.float32, device='cuda')
    new_a_loc_4 = torch.zeros(300, dtype=torch.float32, device='cuda')
    launch_var_len_copy_triton(old_a_start_4, old_a_len_4, old_a_loc_4, new_a_start_4, new_a_loc_4, BLOCK_SIZE)
    results[test_case_4_key] = new_a_loc_4.tolist()
    for (k, v) in results.items():
        print(k, v)
##################################################################################################################################################



import torch

def test_launch_var_len_copy_kernel_triton():
    # Define test input data
    num_arrays = 3
    BLOCK_SIZE = 256

    # Old array start indices
    old_a_start = torch.tensor([0, 100, 300], dtype=torch.int32, device='cuda')

    # Lengths of each array
    old_a_len = torch.tensor([50, 150, 200], dtype=torch.int32, device='cuda')

    # Flattened old array data
    old_a_location = torch.arange(500, dtype=torch.float32, device='cuda')

    # New array start indices
    new_a_start = torch.tensor([0, 60, 260], dtype=torch.int32, device='cuda')

    # Target flattened array for copying
    new_a_location = torch.zeros(500, dtype=torch.float32, device='cuda')

    # Launch the Triton kernel
    launch_var_len_copy_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location)

    # Store results in a dictionary
    results = {}
    for i in range(num_arrays):
        old_start = old_a_start[i].item()
        new_start = new_a_start[i].item()
        length = old_a_len[i].item()
        # Store the result of each test case
        results[f"test_case_{i+1}"] = torch.equal(
            old_a_location[old_start:old_start + length],
            new_a_location[new_start:new_start + length]
        )
    
    return results

result_gold = test_launch_var_len_copy_kernel_triton()
