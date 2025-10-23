import os
from glob import glob
import subprocess
from tqdm import tqdm

path = os.path.dirname(os.path.abspath(__file__))
pattern = os.path.join(path, "test_golden_metrics", "*_perf.py")
files = glob(pattern)

assert len(files) > 0, f"No files found in pattern: {pattern}"

ok_files = ["vector_addition_perf.py","vector_addition_custom_perf.py","var_len_copy_perf.py","triton_matmul_perf.py","triton_linear_activation_perf.py","triton_conv2d_fwd_perf.py","triton_attention_perf.py","token_softmax_llama_perf.py","token_softmax_bloom_perf.py","token_attn_reduceV_perf.py","token_attn_llama2_perf.py","swiglu_triton_perf.py","swiglu_backward_perf.py","streamk_matmul_perf.py","spinning_lock_reduction_perf.py","softmax_reducev_perf.py","sgmv_expand_slice_perf.py","rotary_transform_perf.py","rotary_transform_ops_perf.py","rotary_emb_perf.py","rotary_emb_nopad_perf.py","rope_transform_perf.py","rope_embedding_perf.py","rope_backward_transform_perf.py","rmsnorm_triton_perf.py","rmsnorm_implementation_perf.py","rmsnorm_fused_perf.py","rmsnorm_fused_llama_perf.py","rms_rbe_matmul_perf.py","rms_norm_triton_perf.py","rms_matmul_rbe_perf.py","reversed_cumsum_scalar_perf.py","reversed_cumsum_perf.py","relu_strided_buffer_perf.py","rbe_triton_transform_perf.py","quantize_kv_transform_perf.py","quantize_kv_copy_perf.py","quantize_copy_kv_perf.py","pow_scalar_tensor_perf.py","parallel_retention_attention_perf.py","parallel_attention_perf.py","nested_loops_processing_perf.py","multinomial_sampling_perf.py","mixed_sparse_attention_perf.py","matrix_vector_multip_perf.py","matrix_reduction_perf.py","matmul_triton_autotune_perf.py","matmul_triton2_perf.py","matmul_triton1_perf.py","matmul_tma_perf.py","matmul_persistent_triton_perf.py","matmul_leakyrelu_perf.py","matmul_leakyrelu_fp8_perf.py","matmul_dequantize_perf.py","matmul_dequantize_int4_perf.py","matmul_dequant_int4_perf.py","masked_select_perf.py","masked_add_cuda_perf.py","lora_expand_gemv_perf.py","llama_ff_triton_perf.py","lightning_attention_perf.py","layernorm_fwd_triton_perf.py","layer_norm_welfold_perf.py","layer_norm_triton_perf.py","layer_norm_ops_perf.py","layer_norm_liger_perf.py","layer_norm_fwd_perf.py","l2_norm_bwd_perf.py","kv_cache_filling_perf.py","kv_cache_copy_perf.py","kldiv_triton_perf.py","kldiv_ops_perf.py","kldiv_compute_perf.py","kcache_copy_triton_perf.py","iv_dependent_matmul_perf.py","int_scaled_matmul_perf.py","int8_quantization_perf.py","int8_matmul_quantization_perf.py","int8_matmul_kernel_perf.py","int8_dequant_matmul_perf.py","int4_matmul_perf.py","index_select_cat_perf.py","index_select_bwd_perf.py","geglu_tanh_triton_perf.py","fused_rwkv6_kernel_perf.py","fused_rotary_embedding_perf.py","fused_recurrent_retention_perf.py","fused_recurrent_hgrn_perf.py","fused_recurrent_delta_perf.py","fused_layernorm_triton_perf.py","fused_activation_perf.py","fp4_to_bf16_conversion_perf.py","flash_decode2_phi_perf.py","flash_decode2_llama_perf.py","flash_attn_perf.py","fast_rope_embedding_perf.py","fast_ce_loss_perf.py","f8_conversion_utils_perf.py","embedding_triton_kernel_perf.py","dropout_triton_perf.py","diag_ssm_triton_perf.py","destindex_copy_perf.py","destindex_copy_kv2_perf.py","destindex_copy_kv1_perf.py","dequantize_rowwise_perf.py","dequantize_matmul_perf.py","decay_cumsum_perf.py","cross_entropy_ops_perf.py","cross_entropy2_perf.py","cross_entropy1_perf.py","context_attn_nopad_perf.py","context_attn_mistral_perf.py","context_attn_llama_perf.py","context_attn_fwd_perf.py","context_attn_bloom_perf.py","chunked_cumsum_fwd_perf.py","chunk_retention_ops_perf.py","chunk_linear_attn_perf.py","chunk_gla_simple_perf.py","chunk_gla_fwd_perf.py","chunk_gated_attention_perf.py","chunk_gate_recurrence_perf.py","chunk_delta_fwd_perf.py","chunk_bwd_dqkg_perf.py","cache_transform_triton_perf.py","bmm_optimized_perf.py","bmm_chunk_fwd_perf.py","bmm_chunk_bwd_perf.py","block_sparse_attn_perf.py","bgmv_shrink_kernel_perf.py","bgmv_expand_slice_perf.py","batched_vecmat_mult_perf.py","attn_fwd_triton_perf.py","attn_fwd_causal_perf.py","attention_score_perf.py","attention_llama_perf.py","attention_kernel_perf.py","attention_kernel_aligned_perf.py","attention_fwd_triton3_perf.py","attention_fwd_triton2_perf.py","attention_fwd_triton1_perf.py","attention_forward_triton_perf.py","apply_penalty_perf.py","add_value_perf.py","add_example_perf.py"]

for file in tqdm(files):
    bname = os.path.basename(file)
    if bname in ok_files:
        print(f"Skipping known ok file: {bname}")
        print("--------------------------------------------------")
        continue
    cmd = f"python {file}"

    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"File: {file}, Return Code: {result.returncode}, \n Error: {result.stderr}")
    print("--------------------------------------------------")
