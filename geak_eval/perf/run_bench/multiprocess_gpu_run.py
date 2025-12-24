# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/thunlp/TritonBench - Apache License 2.0
import os
import subprocess
from multiprocessing import Pool, Lock, Value
from tqdm import tqdm
import argparse

def parser():
    parser = argparse.ArgumentParser(description="Run benchmark scripts on multiple GPUs.")
    parser.add_argument("--root_dir", type=str, default="./tmp", help="Directory containing benchmark scripts.")
    args = parser.parse_args()
    return args

gpu_count = 8

# scripts = sorted([f for f in os.listdir(script_dir) if f.endswith(".py")])
# scripts = [os.path.join(script_dir, script) for script in scripts]
# total_scripts = len(scripts)  

progress = Value('i', 0)
progress_lock = Lock()

def run_script(args):
    gpu_id, script = args

    script_name = os.path.basename(script)
    log_file = os.path.join(log_dir, f"{script_name}.log")
    err_file = os.path.join(log_dir, f"{script_name}.err")

    cmd = f"HIP_VISIBLE_DEVICES={gpu_id} python {script}"
    # print(f"Running: {cmd}")

    with open(log_file, "w") as log, open(err_file, "w") as err:
        process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=err)
        process.wait()

    with progress_lock:
        progress.value += 1
        tqdm.write(f"✅ finished {progress.value}/{total_scripts}: {script_name}")


if __name__ == "__main__":
    args = parser()
    script_dir = os.path.join(args.root_dir, "tmp")
    log_dir = os.path.join(args.root_dir, "logs")

    os.makedirs(log_dir, exist_ok=True)

    scripts = sorted([f for f in os.listdir(script_dir) if f.endswith(".py")])
    scripts = [os.path.join(script_dir, script) for script in scripts]
    total_scripts = len(scripts)  

    assert os.path.exists(script_dir), f"Script directory {script_dir} does not exist."

    with Pool(processes=gpu_count) as pool, tqdm(total=total_scripts, desc="Process", ncols=80) as pbar:
        args_list = [(i % gpu_count, scripts[i]) for i in range(total_scripts)]
        
        for _ in pool.imap(run_script, args_list):
            pbar.update(1)

        pool.close()
        pool.join()