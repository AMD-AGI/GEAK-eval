import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swiglu_backward import _swiglu_bwd

import torch
import triton
import triton.language as tl
from geak_eval.data.TritonBench.data.TritonBench_G_v1.swiglu_backward import _swiglu_bwd as _swiglu_bwd_ref
from geak_eval.perf.performance_utils import Performance_Metrics, do_bench_config

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('swiglu_backward', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 14):
            size = 2 ** i
            xy = torch.rand((128, size * 2), dtype=torch.float32).reshape(1, -1)
            dout = torch.rand((128, size), dtype=torch.float32).reshape(1, -1)
            self.input_tensors.append((xy, dout))

    def to_cuda(self, input_tensor):
        xy, dout = input_tensor
        return (xy.cuda(), dout.cuda())

    def call_op(self, input_tensor):
        xy, dout = input_tensor
        return _swiglu_bwd(xy, dout)

    def call_op_ref(self, input_tensor):
        xy, dout = input_tensor
        return _swiglu_bwd_ref(xy, dout)

    def get_gbps(self, input_tensor, runtime):
        xy, dout = input_tensor
        total_bytes = (xy.numel() + dout.numel()) * xy.element_size() * 2  # Read and write
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        xy, dout = input_tensor
        N = dout.numel()
        FLOPS = 5 * N  # Based on the operations in the kernel
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
