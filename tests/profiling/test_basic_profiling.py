#!/usr/bin/env python3
"""
Basic profiling test for quick profiling operations.

This test focuses on basic profiling scenarios:
- Vector creation and initialization
- Basic GPU memory transfers
- Simple operations that can be profiled
"""

import pytest
import cato as ca
import time


class TestBasicProfiling:
    """Test class for basic profiling operations."""
    
    def test_vector_creation_profiling(self):
        """Test profiling of vector creation operations."""
        print("\n=== Testing vector creation profiling ===")

        # Test different vector sizes for profiling
        sizes = [100, 1000, 10000]

        for size in sizes:
            start_time = time.time()
            V = ca.vector(size, ca.zeros())
            creation_time = time.time() - start_time
            
            assert V is not None
            assert V.size() == size
            print(f"Vector size {size}: created in {creation_time:.4f}s")

    def test_gpu_transfer_profiling(self):
        """Test profiling of GPU transfer operations."""
        print("\n=== Testing GPU transfer profiling ===")

        # Create vectors of different sizes
        V1 = ca.vector(1000, ca.ones())
        V2 = ca.vector(10000, ca.random(42))

        # Profile GPU transfer for small vector
        start_time = time.time()
        V1.ensure_on_gpu()
        transfer_time_1 = time.time() - start_time

        assert V1.is_on_gpu()
        print(f"Small vector (1000): GPU transfer in {transfer_time_1:.4f}s")

        # Profile GPU transfer for medium vector
        start_time = time.time()
        V2.ensure_on_gpu()
        transfer_time_2 = time.time() - start_time

        assert V2.is_on_gpu()
        print(f"Medium vector (10000): GPU transfer in {transfer_time_2:.4f}s")

        # Verify that larger vectors take longer (basic profiling validation)
        assert transfer_time_2 > transfer_time_1, "Larger vectors should take longer to transfer"
        print(f"Transfer time ratio: {transfer_time_2/transfer_time_1:.2f}x")

    def test_initialization_function_profiling(self):
        """Test profiling of different initialization functions."""
        print("\n=== Testing initialization function profiling ===")

        # Test different initialization functions
        init_functions = [
            ('zeros', ca.zeros()),
            ('ones', ca.ones()),
            ('constant(2.5)', ca.constant(2.5)),
            ('random(42)', ca.random(42))
        ]

        for name, func in init_functions:
            start_time = time.time()
            V = ca.vector(5000, func)
            init_time = time.time() - start_time

            assert V is not None
            assert V.size() == 5000
            print(f"{name}: initialization in {init_time:.4f}s")

    def test_memory_management_profiling(self):
        """Test profiling of memory management operations."""
        print("\n=== Testing memory management profiling ===")

        V = ca.vector(5000, ca.ones())

        # Profile GPU transfer
        start_time = time.time()
        V.ensure_on_gpu()
        gpu_time = time.time() - start_time
        print(f"GPU transfer: {gpu_time:.4f}s")

        # Profile host transfer
        start_time = time.time()
        V.ensure_on_host()
        host_time = time.time() - start_time
        print(f"Host transfer: {host_time:.4f}s")

        # Profile round-trip
        start_time = time.time()
        V.ensure_on_gpu()
        V.ensure_on_host()
        round_trip_time = time.time() - start_time
        print(f"Round-trip transfer: {round_trip_time:.4f}s")

        # Basic validation
        assert gpu_time > 0, "GPU transfer should take measurable time"
        assert host_time > 0, "Host transfer should take measurable time"
        assert round_trip_time > 0, "Round-trip should take measurable time"
    
    def test_stress_profiling(self):
        """Test profiling under stress conditions."""
        print("\n=== Testing stress profiling ===")

        # Create multiple vectors rapidly
        vectors = []
        start_time = time.time()

        for i in range(10):
            V = ca.vector(1000, ca.constant(i))
            vectors.append(V)
        
        creation_time = time.time() - start_time
        print(f"Created {len(vectors)} vectors in {creation_time:.4f}s")
        
        # Transfer all to GPU
        start_time = time.time()
        for V in vectors:
            V.ensure_on_gpu()

        transfer_time = time.time() - start_time
        print(f"Transferred all vectors to GPU in {transfer_time:.4f}s")

        # Verify all vectors are on GPU
        assert all(V.is_on_gpu() for V in vectors)
        print(f"All {len(vectors)} vectors successfully on GPU")


def main():
    """Main function for standalone execution."""
    print("🚀 Basic Profiling Test - Quick Operations")

    # Create test instance and run all tests
    test_instance = TestBasicProfiling()

    # Run each test method
    test_methods = [
        test_instance.test_vector_creation_profiling,
        test_instance.test_gpu_transfer_profiling,
        test_instance.test_initialization_function_profiling,
        test_instance.test_memory_management_profiling,
        test_instance.test_stress_profiling
    ]

    for test_method in test_methods:
        try:
            test_method()
            print(f"{test_method.__name__} completed successfully")
        except Exception as e:
            print(f"{test_method.__name__} failed: {e}")

    print("\n🎉 Basic profiling test completed!")


if __name__ == "__main__":
    main()
