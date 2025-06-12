# performance_utils_for_pytest.py
import torch
import triton
from typing import Callable, List, Union, Tuple, Dict
import json
import os
import time # For unique filenames if needed

# --- do_bench_config class (can remain largely the same) ---
class do_bench_config():
    def __init__(
        self,
        warm_up: int = 25, # Default values for individual benchmarks
        repetition: int = 100,
        quantiles: List[float] = None,
        return_mode: str = "median"
    ):
        self.warm_up = warm_up
        self.repetition = repetition
        self.quantiles = quantiles if quantiles is not None else [0.5, 0.8, 0.2]
        self.return_mode = return_mode

# --- Global Collector for results ---
# This is a common way to gather data from pytest runs.
# It needs to be managed carefully, especially with parallel pytest (xdist).
# For non-parallel pytest, it's simpler.
# Let's assume non-parallel pytest for now for simplicity of collection.
# If using pytest-xdist, inter-process communication for collection is needed.

PYTEST_BENCHMARK_RESULTS = {} # Key: op_name, Value: list of result dicts

def add_benchmark_result(op_name: str, result_dict: Dict):
    if op_name not in PYTEST_BENCHMARK_RESULTS:
        PYTEST_BENCHMARK_RESULTS[op_name] = []
    PYTEST_BENCHMARK_RESULTS[op_name].append(result_dict)

def save_all_benchmark_results(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for op_name, results_list in PYTEST_BENCHMARK_RESULTS.items():
        if results_list: # Only save if there are results
            file_name = op_name + ".json" # e.g., add_kernel.json
            file_path = os.path.join(output_dir, file_name)
            try:
                with open(file_path, 'w', encoding='utf8') as f:
                    json.dump(results_list, f, indent=4, ensure_ascii=False)
                print(f"Benchmark results for {op_name} saved to: {file_path}")
            except IOError as e:
                print(f"Error saving results for {op_name} to {file_path}: {e}")
    PYTEST_BENCHMARK_RESULTS.clear() # Clear after saving

# --- Simplified Benchmarking Helper ---
class PytestBenchmarker:
    def __init__(self, op_callable: Callable, op_name: str, config: do_bench_config = None):
        self.op_callable = op_callable # This will be the kernel launch lambda
        self.op_name = op_name # e.g., "add_kernel" or "test_add_SIZE_BLOCK_SIZE_dtype"
        self.config = config if config else do_bench_config()

    def run_benchmark(self, current_params_dict: Dict,
                      gbps_calculator: Callable = None, # (inputs, ms) -> gbps
                      tflops_calculator: Callable = None # (inputs, ms) -> tflops
                     ):
        """
        Runs the benchmark for the op_callable (which should be a lambda with inputs already bound).
        current_params_dict: Dictionary describing the current pytest parameters.
        gbps_calculator: A function that takes (original_inputs_tuple, ms) and returns GB/s.
        tflops_calculator: A function that takes (original_inputs_tuple, ms) and returns TFLOPS.
        """
        try:
            ms, min_ms, max_ms = triton.testing.do_bench(
                self.op_callable,
                warmup=self.config.warm_up,
                rep=self.config.repetition,
                quantiles=self.config.quantiles,
                return_mode=self.config.return_mode
            )

            gbps = "N/A"
            if gbps_calculator:
                try:
                    # gbps_calculator needs access to the input tensors that define the problem size
                    # The caller of run_benchmark will need to provide these if they are not part of current_params_dict
                    # For simplicity, let's assume current_params_dict contains enough info, or calculator can access them
                    gbps = gbps_calculator(current_params_dict, ms)
                except Exception as e_gbps:
                    print(f"Warning: GB/s calculation failed for {self.op_name} with params {current_params_dict}: {e_gbps}")


            tflops = "N/A"
            if tflops_calculator:
                try:
                    tflops = tflops_calculator(current_params_dict, ms)
                except Exception as e_tflops:
                    print(f"Warning: TFLOPS calculation failed for {self.op_name} with params {current_params_dict}: {e_tflops}")


            result = {
                "params": current_params_dict, # From pytest.mark.parametrize
                "ms": round(ms, 4),
                "min_ms": round(min_ms, 4),
                "max_ms": round(max_ms, 4),
                "GB/s": round(gbps, 2) if isinstance(gbps, (float, int)) else gbps,
                "TFLOPS": round(tflops, 2) if isinstance(tflops, (float, int)) else tflops,
            }
            # Use a unique op_name for each kernel/function being benchmarked
            add_benchmark_result(self.op_name, result)
            # print(f"Benchmarked {self.op_name} with {current_params_dict}: {ms:.4f} ms")
            return result

        except Exception as e:
            print(f"Error during benchmark for {self.op_name} with params {current_params_dict}: {e}")
            error_result = {
                "params": current_params_dict,
                "error": str(e)
            }
            add_benchmark_result(self.op_name, error_result)
            return error_result