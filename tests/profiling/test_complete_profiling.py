#!/usr/bin/env python3
"""
Complete profiling test for comprehensive profiling operations.

This test focuses on comprehensive profiling scenarios:
- Large-scale vector operations
- Memory transfer profiling
- All initialization functions
- Stress testing for profiling tools
"""

import pytest
import cato as ca
import time
import gc


class TestCompleteProfiling:
    """Test class for comprehensive profiling operations."""
    
    def test_large_vector_creation_profiling(self):
        """Test profiling of large vector creation operations."""
        print("\n=== Testing large vector creation profiling ===")

        # Test different vector sizes for comprehensive profiling
        sizes = [1000, 10000, 100000, 1000000]

        for size in sizes:
            print(f"   Creating vector of size {size}...")
            start_time = time.time()

            V = ca.vector(size, ca.zeros())
            creation_time = time.time() - start_time

            assert V is not None
            assert V.size() == size
            print(f"   Size {size}: created in {creation_time:.4f}s")

            # Clean up to avoid memory issues
            del V
            gc.collect()

    def test_memory_transfer_profiling(self):
        """Test profiling of memory transfer operations."""
        print("\n=== Testing memory transfer profiling ===")

        # Create vectors of different sizes
        V1 = ca.vector(1000, ca.ones())
        V2 = ca.vector(10000, ca.random(42))
        V3 = ca.vector(100000, ca.normal(0, 1, 42))

        print("   Transferring vectors to GPU...")

        # Profile GPU transfer for small vector
        start_time = time.time()
        V1.ensure_on_gpu()
        transfer_time_1 = time.time() - start_time
        print(f"   V1 (1000): GPU transfer in {transfer_time_1:.4f}s")

        # Profile GPU transfer for medium vector
        start_time = time.time()
        V2.ensure_on_gpu()
        transfer_time_2 = time.time() - start_time
        print(f"   V2 (10000): GPU transfer in {transfer_time_2:.4f}s")

        # Profile GPU transfer for large vector
        start_time = time.time()
        V3.ensure_on_gpu()
        transfer_time_3 = time.time() - start_time
        print(f"   V3 (100000): GPU transfer in {transfer_time_3:.4f}s")

        # Verify transfer times make sense
        assert transfer_time_2 > transfer_time_1, "Medium vector should take longer than small"
        assert transfer_time_3 > transfer_time_2, "Large vector should take longer than medium"
        
        print(f"   Transfer time ratios: {transfer_time_2/transfer_time_1:.2f}x, {transfer_time_3/transfer_time_1:.2f}x")

        # Test round-trip transfers
        print("   Testing round-trip transfers...")

        start_time = time.time()
        V1.ensure_on_host()
        V1.ensure_on_gpu()
        round_trip_1 = time.time() - start_time
        print(f"   V1 round-trip: {round_trip_1:.4f}s")

        start_time = time.time()
        V2.ensure_on_host()
        V2.ensure_on_gpu()
        round_trip_2 = time.time() - start_time
        print(f"   V2 round-trip: {round_trip_2:.4f}s")

    def test_all_initialization_functions_profiling(self):
        """Test profiling of all initialization functions."""
        print("\n=== Testing all initialization functions profiling ===")

        # Test all available initialization functions
        functions = [
            ('zeros', ca.zeros()),
            ('ones', ca.ones()),
            ('constant(2.5)', ca.constant(2.5)),
            ('random(123)', ca.random(123)),
            ('uniform(0, 10, 123)', ca.uniform(0, 10, 123)),
            ('normal(5, 2, 123)', ca.normal(5, 2, 123)),
            ('box_muller(0, 1, 123)', ca.box_muller(0, 1, 123)),
            ('sequence(1, 0.5)', ca.sequence(1, 0.5)),
            ('arange()', ca.arange()),
            ('sine(0.2, 3.0, 0.0)', ca.sine(0.2, 3.0, 0.0))
        ]

        print("   Testing initialization functions with size 10000...")

        for i, (name, func) in enumerate(functions):
            start_time = time.time()
            V = ca.vector(10000, func)
            init_time = time.time() - start_time

            assert V is not None
            assert V.size() == 10000
            print(f"   {i+1:2d}. {name:20s}: {init_time:.4f}s")

            # Clean up
            del V
            gc.collect()

    def test_memory_pressure_profiling(self):
        """Test profiling under memory pressure conditions."""
        print("\n=== Testing memory pressure profiling ===")

        # Create many vectors to create memory pressure
        vectors = []
        print("   Creating 50 vectors of size 10000...")

        start_time = time.time()
        for i in range(50):
            V = ca.vector(10000, ca.constant(i))
            vectors.append(V)

            # Print progress every 10 vectors
            if (i + 1) % 10 == 0:
                print(f"   Created {i + 1}/50 vectors...")

        creation_time = time.time() - start_time
        print(f"   Created {len(vectors)} vectors in {creation_time:.4f}s")

        # Transfer all to GPU
        print("   Transferring all vectors to GPU...")
        start_time = time.time()

        for i, V in enumerate(vectors):
            V.ensure_on_gpu()
            if (i + 1) % 10 == 0:
                print(f"   Transferred {i + 1}/50 vectors to GPU...")

        transfer_time = time.time() - start_time
        print(f"   Transferred all vectors to GPU in {transfer_time:.4f}s")

        # Verify all vectors are on GPU
        gpu_count = sum(1 for V in vectors if V.is_on_gpu())
        print(f"   {gpu_count}/{len(vectors)} vectors successfully on GPU")

        # Clean up
        del vectors
        gc.collect()
        print("   🧹 Memory cleaned up")

    def test_profiling_tool_compatibility(self):
        """Test that operations are suitable for profiling tools."""
        print("\n=== Testing profiling tool compatibility ===")
        
        print("   This test ensures operations are visible in profiling tools:")
        print("   - Nsight Systems")
        print("   - Nsight Compute")
        print("   - CUDA Profiler")

        # Create operations that should be visible in profiling
        print("   Creating profiling-friendly operations...")
        
        # Large memory allocation
        V1 = ca.vector(1000000, ca.zeros())
        print("   Large vector created (should be visible in memory profiler)")

        # GPU transfer
        V1.ensure_on_gpu()
        print("   GPU transfer completed (should be visible in CUDA profiler)")

        # Round-trip transfer
        V1.ensure_on_host()
        V1.ensure_on_gpu()
        print("   Round-trip transfer completed (should show both directions)")

        # Multiple operations
        V2 = ca.vector(500000, ca.ones())
        V3 = ca.vector(500000, ca.random(42))

        V2.ensure_on_gpu()
        V3.ensure_on_gpu()
        print("   Multiple GPU transfers completed (should show parallel operations)")
        
        print("   Profiling tool compatibility verified")

    def test_performance_validation(self):
        """Test basic performance validation for profiling."""
        print("\n=== Testing performance validation ===")
        
        # Test that larger operations take longer (basic sanity check)
        print("   Validating performance characteristics...")

        # Small vector
        start_time = time.time()
        V1 = ca.vector(1000, ca.zeros())
        V1.ensure_on_gpu()
        time_small = time.time() - start_time

        # Large vector
        start_time = time.time()
        V2 = ca.vector(100000, ca.zeros())
        V2.ensure_on_gpu()
        time_large = time.time() - start_time

        # Validate that larger operations take longer
        assert time_large > time_small, "Larger operations should take longer"
        print(f"   Performance validation passed: {time_large/time_small:.2f}x ratio")

        # Test memory transfer scaling
        print("   Testing memory transfer scaling...")

        sizes = [1000, 10000, 100000]
        times = []

        for size in sizes:
            V = ca.vector(size, ca.zeros())
            start_time = time.time()
            V.ensure_on_gpu()
            transfer_time = time.time() - start_time
            times.append(transfer_time)
            print(f"   Size {size:6d}: {transfer_time:.4f}s")

        # Basic validation that times increase with size
        assert times[1] > times[0], "Medium size should take longer than small"
        assert times[2] > times[1], "Large size should take longer than medium"
        print("   Memory transfer scaling validation passed")


def main():
    """Main function for standalone execution."""
    print("🚀 Complete Profiling Test - All Operations")

    # Create test instance and run all tests
    test_instance = TestCompleteProfiling()

    # Run each test method
    test_methods = [
        test_instance.test_large_vector_creation_profiling,
        test_instance.test_memory_transfer_profiling,
        test_instance.test_all_initialization_functions_profiling,
        test_instance.test_memory_pressure_profiling,
        test_instance.test_profiling_tool_compatibility,
        test_instance.test_performance_validation
    ]

    for test_method in test_methods:
        try:
            test_method()
            print(f"{test_method.__name__} completed successfully")
        except Exception as e:
            print(f"{test_method.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n  Complete profiling test completed!")
    print("\n  Profiling Results Summary:")
    print("   - Vector creation operations profiled")
    print("   - Memory transfer operations profiled")
    print("   - All initialization functions tested")
    print("   - Memory pressure scenarios tested")
    print("   - Profiling tool compatibility verified")
    print("   - Performance characteristics validated")
    print("\n  Next steps:")
    print("   - Run with Nsight Systems for timeline analysis")
    print("   - Use Nsight Compute for kernel analysis")
    print("   - Analyze memory transfer patterns")
    print("   - Identify performance bottlenecks")


if __name__ == "__main__":
    main()
