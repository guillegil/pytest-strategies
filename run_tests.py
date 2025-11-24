#!/usr/bin/env python3
"""
Run all tests for pytest-strategies, including unit, integration, and corner cases.
"""
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and print status."""
    print(f"Running: {description}...")
    print(f"  $ {command}")
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"  ✅ Passed in {duration:.2f}s\n")
        return True
    else:
        print(f"  ❌ Failed in {duration:.2f}s\n")
        return False

def main():
    print("========================================================")
    print("       Running pytest-strategies Test Suite             ")
    print("========================================================\n")

    failures = []

    # 1. Unit Tests
    if not run_command("pytest tests/unittests -v", "Unit Tests"):
        failures.append("Unit Tests")

    # 2. Integration Tests (Default nsamples=10)
    if not run_command("pytest tests/integration -v", "Integration Tests (Default)"):
        failures.append("Integration Tests (Default)")

    # 3. Sequence Testing (nsamples="auto")
    # This verifies the exhaustive generation logic
    if not run_command("pytest tests/integration/test_sequence_integration.py --nsamples=auto -v", "Sequence Tests (Auto Mode)"):
        failures.append("Sequence Tests (Auto Mode)")

    # 4. Legacy Strategy Compatibility
    # Verify legacy tuple strategies still work with default nsamples
    # (The fix we applied handles string input)
    if not run_command("pytest tests/integration/test_full_integration.py -v", "Legacy Strategy Compatibility"):
        failures.append("Legacy Strategy Compatibility")

    # 5. Example Scripts
    # Verify examples run correctly
    if not run_command("pytest examples/sequence_example.py --nsamples=auto -v", "Example: Sequence (Auto)"):
        failures.append("Example: Sequence (Auto)")
        
    if not run_command("pytest examples/sequence_example.py --nsamples=5 -v", "Example: Sequence (Random)"):
        failures.append("Example: Sequence (Random)")

    # 6. Test Values Examples
    # Verify test mode runs only test vectors
    if not run_command("pytest examples/test_values_example.py --vector-mode=test -v", "Example: Test Values (Test Mode)"):
        failures.append("Example: Test Values (Test Mode)")
        
    if not run_command("pytest examples/test_values_example.py --nsamples=5 -v", "Example: Test Values (All Mode)"):
        failures.append("Example: Test Values (All Mode)")

    # Summary
    print("========================================================")
    if not failures:
        print("🎉 ALL TESTS PASSED! Ready for release.")
        sys.exit(0)
    else:
        print(f"❌ {len(failures)} TEST SUITES FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)

if __name__ == "__main__":
    main()
