import os
from .perf.efficiency import PerformanceEvalTBG
from .constants import TBG_PERF_GOLD_ROOT, TBG_DATA_ROOT, NATIVE_PERF_GOLD_ROOT

def initialize_performance_eval_tb():
    perf_evaluator = PerformanceEvalTBG()
    perf_evaluator.ref_folder = NATIVE_PERF_GOLD_ROOT
    print(f"Creating performance evaluation folder at {TBG_PERF_GOLD_ROOT}")
    perf_evaluator(exec_folder=TBG_DATA_ROOT, gen_perf_folder=TBG_PERF_GOLD_ROOT, golden_metrics_folder=NATIVE_PERF_GOLD_ROOT)

if __name__ == "__main__":
    initialize_performance_eval_tb()
