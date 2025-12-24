Analyzing 31 file(s) for parametrized tests...

Generated 22 unique block configurations for multreduce matmul.
📁 ./triton_multreduce_matmul_kernel.py: 80 test cases
  └── test_matmul: 14 cases
  └── test_performance: 66 cases
📁 ./test_triton_swizzle2d.py: 16 test cases
  └── test_performance: 16 cases
📁 ./test_triton_sort.py: 128 test cases
  └── test_sort: 32 cases
  └── test_performance: 96 cases
📁 ./test_triton_flip.py: 40 test cases
  └── test_flip: 16 cases
  └── test_performance: 24 cases
📁 ./test_tma_store_gemm.py: 19 test cases
  └── test_tma_load_store: 8 cases
  └── test_performance: 11 cases
📁 ./test_reverse_range.py: 9 test cases
  └── test_performance: 9 cases
📁 ./test_random_int.py: 136 test cases
  └── test_randint: 72 cases
  └── test_performance: 64 cases
📁 ./test_randn.py: 120 test cases
  └── test_rand: 16 cases
  └── test_performance: 104 cases
📁 ./test_matmul_MXFP.py: 8 test cases
  └── test_pipeline_matmul: 2 cases
  └── test_performance: 6 cases
📁 ./test_load_reduce.py: 42 test cases
  └── test_performance: 42 cases
Warning: Could not load module ./test_kernel_sub.py: cannot import name 'AttrsDescriptor' from 'triton.backends.compiler' (/opt/conda/envs/py_3.12/lib/python3.12/site-packages/triton/backends/compiler.py)
📁 ./test_kernel_sub.py: No parametrized tests found
Warning: Could not load module ./test_kernel_dot.py: cannot import name 'AttrsDescriptor' from 'triton.backends.compiler' (/opt/conda/envs/py_3.12/lib/python3.12/site-packages/triton/backends/compiler.py)
📁 ./test_kernel_dot.py: No parametrized tests found
📁 ./test_iv_dependent_matmul.py: 2105 test cases
  └── test_iv_dependent_matmul: 5 cases
  └── test_performance: 2100 cases
📁 ./test_gemm_no_scf.py: No parametrized tests found
Generated 516 unique performance test configurations for gemm_fusion.
📁 ./test_gemm_fusion.py: 516 test cases
  └── test_performance: 516 cases
📁 ./test_flashattention_fwd.py: 12 test cases
  └── test_op: 6 cases
  └── test_performance: 6 cases
Generated 150 unique performance test configurations for chained_matmul.
📁 ./test_chained_matmul.py: 150 test cases
  └── test_performance: 150 cases
📁 ./test_chained_dot_fp8.py: 24 test cases
  └── test_chained_dot: 4 cases
  └── test_performance: 20 cases
📁 ./test_cast_matmul.py: 54 test cases
  └── test_cast_matmul: 36 cases
  └── test_performance: 18 cases
📁 ./test_block_pointer_matmul.py: 26 test cases
  └── test_block_ptr_matmul_no_scf: 6 cases
  └── test_performance: 20 cases
📁 ./test_block_copy.py: 130 test cases
  └── test_block_copy: 90 cases
  └── test_performance: 40 cases
📁 ./test_batched_vecmat.py: 30 test cases
  └── test_performance: 30 cases
📁 ./test_add_kernel.py: 6 test cases
  └── test_add: 2 cases
  └── test_performance: 4 cases
📁 ./softmax.py: 31 test cases
  └── test_softmax: 10 cases
  └── test_performance: 21 cases
📁 ./rmsnorm_fwd.py: 182 test cases
  └── test_rmsnorm: 56 cases
  └── test_performance: 126 cases
📁 ./rmsnorm_bwd.py: 82 test cases
  └── test_rmsnorm: 40 cases
  └── test_performance: 42 cases
📁 ./naive_softmax.py: 31 test cases
  └── test_softmax: 10 cases
  └── test_performance: 21 cases
📁 ./multreduce_matmul_dot_kernel.py: 38 test cases
  └── test_matmul: 14 cases
  └── test_performance: 24 cases
📁 ./moe_gemm.py: 108 test cases
  └── test_correctness: 20 cases
  └── test_correctness_fp8: 36 cases
  └── test_correctness_int8_w8a16: 18 cases
  └── test_correctness_int8_w8a8: 18 cases
  └── test_performance: 16 cases
📁 ./layernorm.py: 35 test cases
  └── test_layernorm: 11 cases
  └── test_performance: 24 cases
📁 ./gemm.py: 517 test cases
  └── test_correctness: 484 cases
  └── test_performance: 33 cases

📊 Summary:
   Files analyzed: 31
   Files with parametrized tests: 28
   Total parametrized test cases: 4675


| Filename | Tests Correctness | Performance |
|----------|-------------------------|-------------|
| triton_multreduce_matmul_kernel.py | test_matmul: 14 | 66 |
| test_triton_swizzle2d.py | - | 16 |
| test_triton_sort.py | test_sort: 32 | 96 |
| test_triton_flip.py | test_flip: 16 | 24 |
| test_tma_store_gemm.py | test_tma_load_store: 8 | 11 |
| test_reverse_range.py | - | 9 |
| test_random_int.py | test_randint: 72 | 64 |
| test_randn.py | test_rand: 16 | 104 |
| test_matmul_MXFP.py | test_pipeline_matmul: 2 | 6 |
| test_load_reduce.py | - | 42 |
| test_iv_dependent_matmul.py | test_iv_dependent_matmul: 5 | 2100 |
| test_gemm_fusion.py | - | 516 |
| test_flashattention_fwd.py | test_op: 6 | 6 |
| test_chained_matmul.py | - | 150 |
| test_chained_dot_fp8.py | test_chained_dot: 4 | 20 |
| test_cast_matmul.py | test_cast_matmul: 36 | 18 |
| test_block_pointer_matmul.py | test_block_ptr_matmul_no_scf: 6 | 20 |
| test_block_copy.py | test_block_copy: 90 | 40 |
| test_batched_vecmat.py | - | 30 |
| test_add_kernel.py | test_add: 2 | 4 |
| softmax.py | test_softmax: 10 | 21 |
| rmsnorm_fwd.py | test_rmsnorm: 56 | 126 |
| rmsnorm_bwd.py | test_rmsnorm: 40 | 42 |
| naive_softmax.py | test_softmax: 10 | 21 |
| multreduce_matmul_dot_kernel.py | test_matmul: 14 | 24 |
| moe_gemm.py | test_correctness: 20, test_correctness_fp8: 36, test_correctness_int8_w8a16: 18, test_correctness_int8_w8a8: 18 | 16 |
| layernorm.py | test_layernorm: 11 | 24 |
| gemm.py | test_correctness: 484 | 33 |