import torch
import triton
import triton.language as tl

from typing import Callable
import json
import os
import random

def get_random_choice(item_list):
    return random.choice(item_list)

class do_bench_config():
    def __init__(
            self,
            warm_up=25,
            repetition=100,
            grad_to_none=None,
            quantiles=[0.5, 0.8, 0.2],
            return_mode="median"
    ):
        self.warm_up = warm_up
        self.repetition = repetition
        self.grad_to_none = grad_to_none
        self.quantiles = quantiles
        self.return_mode = return_mode

class Performance_Metrics:
    def __init__(
            self,
            op_name,
            dtype=None,
            is_backward=False,
            **kwargs
    ):
        self.op_name = op_name
        self.ref_op_name = op_name + '_ref'
        self.dtype = dtype
        if is_backward:
            self.op_name += 'backward'
        self.kwargs = kwargs

        self.input_tensors = []
        self.do_bench_config = do_bench_config()

    def get_input_tensors(self):
        raise NotImplementedError("You must implement this method to get input tensors")

    def to_cuda(self, input_tensor):
        raise NotImplementedError("You must implement this method to get input tensors")
    
    def call_op(self, input_tensor):
        raise NotImplementedError("You must implement this method to call the op")

    def call_op_ref(self, input_tensor):
        raise NotImplementedError("You must implement this method to call the reference op")

    def get_do_bench_config(self, warmup=None, rep=None):
        if warmup != None and rep != None:
            self.do_bench_config = do_bench_config(
                warm_up=warmup,
                repetition=rep,
            )
            return

        if self.input_tensors == []:
            raise NotImplementedError("You must implement this method to get input_tensors")
        
        previous_ms = None
        epsilon = 1e-4
        stable_count = 0
        max_stable_count = 3
        input_tensor = self.to_cuda(self.input_tensors[-1])

        for t in range(1, 11):
            warmup = 100 * t
            rep = 1000 * t
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: self.call_op(input_tensor),
                warmup=warmup,
                rep=rep,
                quantiles=[0.5, 0.8, 0.2],
                return_mode="median"
            )

            print("warmup time:", warmup, "rep time:", rep, "runtime:", ms)

            if previous_ms is not None:
                relative_change = abs(ms - previous_ms) / abs(previous_ms) if previous_ms != 0 else float('inf')

                if relative_change < epsilon:
                    stable_count += 1
                else:
                    stable_count = 0
            
            if stable_count >= max_stable_count:
                print(f"MS stabilized with warmup={warmup} and rep={rep}")
                self.do_bench_config = do_bench_config(
                    warm_up=warmup,
                    repetition=rep,
                )
                return

            previous_ms = ms
        
        print("MS did not stabilize. Returning last config.")
        self.do_bench_config = do_bench_config(
            warm_up=warmup,
            repetition=rep,
        )
        return 
        # raise NotImplementedError("You must implement this method to make the runtime stable")

    def get_runtime(self, op: Callable):
        ms, min_ms, max_ms = triton.testing.do_bench(
            op,
            warmup=self.do_bench_config.warm_up,
            rep=self.do_bench_config.repetition,
            quantiles=self.do_bench_config.quantiles,
            return_mode=self.do_bench_config.return_mode
        )
        return ms
    
    def get_gbps(self, input_tensor, runtime):
        raise NotImplementedError("You must implement this method to get the method to calculate GBPS")

    def get_tflops(self, input_tensor, runtime):
        raise NotImplementedError("You must implement this method to get the method to calculate TFLOPS")

    def check_close(self, a, b, rtol=1e-05, atol=1e-08):
        if isinstance(a, (list, tuple)):
            return all(self.check_close(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b))
        if isinstance(a, dict):
            return all(key in b and self.check_close(a[key], b[key], rtol=rtol, atol=atol) for key in a)
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.allclose(a, b, rtol=rtol, atol=atol)
        return a == b

    def get_num_elements(self, input_tensor):
        if isinstance(input_tensor, (list, tuple)):
            return sum(self.get_num_elements(x) for x in input_tensor)
        if isinstance(input_tensor, dict):
            return sum(self.get_num_elements(v) for v in input_tensor.values())
        if isinstance(input_tensor, torch.Tensor):
            return input_tensor.numel()
        return 1

    def run_benchmark(self):
        results = []
        perf = []
        perf_ref = []
        for input_tensor_ in self.input_tensors:
            try:
                input_tensor = self.to_cuda(input_tensor_)
                # print(input_tensor)
                op = lambda : self.call_op(input_tensor)            
                op_ref = lambda : self.call_op_ref(input_tensor)
                
                ## Keep dummy initial calls to converge to optimal triton autotune configs regardless it exists or not!
                output = self.call_op(input_tensor)
                output_ref = self.call_op_ref(input_tensor)                

                ## The following calls should be using the optimal triton autotune configs for given inputs!
                output = self.call_op(input_tensor)
                output_ref = self.call_op_ref(input_tensor)
                
                if not self.check_close(output, output_ref, rtol=1e-3, atol=1e-3):
                    print(f"Failed to run benchmark for input tensor. Error: {e}")
                    return False, f"Output mismatch between the operation and its reference implementation for input tensor shape"

                # Randomly choose which operation to run first
                # to avoid any bias in the performance measurement                
                if get_random_choice([0, 1]) == 0:
                    ms = self.get_runtime(op)
                    ms_ref = self.get_runtime(op_ref)
                else:
                    ms_ref = self.get_runtime(op_ref)
                    ms = self.get_runtime(op)
                
                gbps = self.get_gbps(input_tensor, ms)
                tflops = self.get_tflops(input_tensor, ms)
                result = {
                    "input_size": self.get_num_elements(input_tensor_),
                    "ms": ms,
                    "ms_ref": ms_ref,
                    "GB/s": gbps,
                    "TFLOPS": tflops
                }
                # print(result)
                results.append(result)
                perf.append(ms)
                perf_ref.append(ms_ref)
            except Exception as e:
                print(f"Failed to run benchmark for input tensor. Error: {e}")
                return False, f"Failed to run benchmark for an input tensor shape due to {e}"
            input_tensor = None

        ## calculate average performance
        if perf and perf_ref:
            avg_perf = sum(perf_ref) / sum(perf)

        results.append({
            "speedup": avg_perf
        })

        print(f"```json\n{json.dumps(results, indent=4)}\n```")

        return True, f"```json\n{json.dumps(results, indent=4)}\n```"
