import torch

def chunk_fwd_h_fn(k, v, d, use_initial_state=False, store_final_state=False, initial_state=None, final_state=None):
    h = torch.sum(k) + torch.sum(v) + torch.sum(d)
    v_new = torch.mul(v, d)
    if use_initial_state and initial_state is not None:
        h += torch.sum(initial_state)
    if store_final_state and final_state is not None:
        v_new += torch.sum(final_state)
    return (h, v_new)
def test_chunk_fwd_h():
    results = {}
    (BT, BK, BV) = (2, 4, 4)
    k = torch.randn((BT, BK), device='cuda', dtype=torch.float32)
    v = torch.randn((BT, BV), device='cuda', dtype=torch.float32)
    d = torch.randn((BT, BV), device='cuda', dtype=torch.float32)
    init_state = torch.randn((1, BV), device='cuda', dtype=torch.float32)
    fin_state = torch.randn((1, BV), device='cuda', dtype=torch.float32)
    (h1, v_new1) = chunk_fwd_h_fn(k, v, d, use_initial_state=False, store_final_state=False)
    results['test_case_1'] = (h1, v_new1)
    (h2, v_new2) = chunk_fwd_h_fn(k, v, d, use_initial_state=True, store_final_state=False, initial_state=init_state)
    results['test_case_2'] = (h2, v_new2)
    (h3, v_new3) = chunk_fwd_h_fn(k, v, d, use_initial_state=False, store_final_state=True, final_state=fin_state)
    results['test_case_3'] = (h3, v_new3)
    (h4, v_new4) = chunk_fwd_h_fn(k, v, d, use_initial_state=True, store_final_state=True, initial_state=init_state, final_state=fin_state)
    results['test_case_4'] = (h4, v_new4)
    print(results)
##################################################################################################################################################



import torch

# Test function for chunk_fwd_h_fn
def test_chunk_fwd_h_fn():
    B, H, T, K, V = 2, 4, 128, 64, 64  # Example dimensions
    BT = 32  # Block size for T dimension

    k = torch.randn(B, H, K, T, dtype=torch.float32, device='cuda')
    w = torch.randn(B, H, T, K, dtype=torch.float32, device='cuda')
    u = torch.randn(B, H, T, V, dtype=torch.float32, device='cuda')

    results = {}

    # Test without initial and final states
    h, v_new = chunk_fwd_h_fn(k, w, u, BT, initial_state=None, final_state=None)
    results['test_case_1'] = (h.shape, v_new.shape)

    # Test with initial and final states
    initial_state = torch.zeros(B, H, K, V, dtype=torch.float32, device='cuda')
    final_state = torch.zeros(B, H, K, V, dtype=torch.float32, device='cuda')
    h, v_new = chunk_fwd_h_fn(k, w, u, BT, initial_state=initial_state, final_state=final_state)
    results['test_case_2'] = (h.shape, v_new.shape)

    return results

# Run tests
result_gold = test_chunk_fwd_h_fn()
