import argparse
import os
import sys
import tempfile
import shutil

# This assumes the script is run from the root of the TB-eval-OE project,
# so it can correctly import the library components.
try:
    from geak_eval.evaluators.interface import get_evaluators
    from geak_eval.constants import Names
except ImportError:
    print("❌ ERROR: Could not import geak_eval modules.")
    print("Please make sure you run this script from the root of the 'TB-eval-OE' project directory.")
    sys.exit(1)

def check_kernel_correctness(program_path: str, atol: float, rtol: float, verbose: bool):
    """
    Checks the numerical correctness of a Triton kernel file using the geak_eval library's
    ROCm evaluator.

    Args:
        program_path (str): Path to the Python file containing the Triton kernel code.
        atol (float): Absolute tolerance for the correctness check.
        rtol (float): Relative tolerance for the correctness check.
        verbose (bool): If True, prints detailed error messages.
    """
    if not os.path.exists(program_path):
        print(f"❌ ERROR: File not found at '{program_path}'")
        sys.exit(1)

    print("=" * 80)
    print(f"🔬 Starting correctness check for: {os.path.basename(program_path)}")
    print(f"   Using Tolerance: atol={atol}, rtol={rtol}")
    print("=" * 80)

    # Instantiate the ROCm evaluator from the geak_eval library
    try:
        evaluator = get_evaluators['rocm']()
        print("✅ Successfully instantiated TestAllCloseEvaluatorROCm.")
    except Exception as e:
        print(f"❌ ERROR: Failed to instantiate ROCm evaluator: {e}")
        sys.exit(1)

    # Read the content of the kernel file
    with open(program_path, 'r') as f:
        kernel_code = f.read()

    # The evaluator needs temporary directories to work with
    with tempfile.TemporaryDirectory() as temp_dir:
        log_root = os.path.join(temp_dir, 'logs')
        exec_root = os.path.join(temp_dir, 'exec')
        os.makedirs(log_root, exist_ok=True)
        os.makedirs(exec_root, exist_ok=True)
        print(f"📦 Created temporary directories for evaluation in: {temp_dir}")

        # The 'fname' is the basename, which the evaluator uses to find the
        # corresponding ground-truth testing harness.
        fname = os.path.basename(program_path)

        print("\n🚀 Executing the correctness check via evaluator...")
        # The evaluator returns: (call_status, exec_status, stdout, stderr)
        call_status, exec_status, stdout, stderr = evaluator.execute(
            code=kernel_code,
            log_root=log_root,
            exec_root=exec_root,
            fname=fname,
            atol=atol,
            rtol=rtol,
            verbose=verbose
        )

        # --- Report the results ---
        print("-" * 80)
        print(f"📊 Results:")
        print(f"   - Kernel Compilation & Call Successful (call_status): {call_status}")
        print(f"   - Numerical Correctness Passed (exec_status):    {exec_status}")
        print("-" * 80)
        print("📄 Standard Output from Evaluator:",stdout)
        print("Start of Standard Error from Evaluator:",stderr)
        if call_status and exec_status:
            print("✅✅✅ CORRECTNESS CHECK PASSED ✅✅✅")
            print("The kernel is numerically correct across all test cases.")
        else:
            print("❌❌❌ CORRECTNESS CHECK FAILED ❌❌❌")
            if not call_status:
                print("   Reason: The kernel failed to compile or execute (call failed).")
            elif not exec_status:
                print("   Reason: The kernel executed but produced numerically incorrect results.")
            
            print("\n--- Error Details ---")
            # The detailed error message is in the 'stderr' return value
            print(stderr or "No detailed error message was returned.")

    print("=" * 80)
    print("✨ Check complete.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to check the numerical correctness of a Triton kernel file using the geak_eval ROCm evaluator.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the Python file containing the Triton kernel(s) to test."
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for torch.allclose(). Default: 1e-2"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-1,
        help="Relative tolerance for torch.allclose(). Default: 1e-1"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output from the evaluator."
    )
    args = parser.parse_args()

    check_kernel_correctness(args.file, args.atol, args.rtol, args.verbose)
