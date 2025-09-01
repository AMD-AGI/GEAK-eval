import os
from glob import glob
import subprocess
from tqdm import tqdm

path = os.path.dirname(os.path.abspath(__file__))
pattern = os.path.join(path, "test_golden_metrics", "*_perf.py")
files = glob(pattern)

assert len(files) > 0, f"No files found in pattern: {pattern}"

for file in tqdm(files):
    cmd = f"python {file}"

    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"File: {file}, Return Code: {result.returncode}, \n Error: {result.stderr}")
    print("--------------------------------------------------")
