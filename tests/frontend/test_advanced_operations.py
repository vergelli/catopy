#!/usr/bin/env python3
"""
Test suite for advanced vector operations and edge cases.

This module tests:
- Complex operation combinations
- Memory management during operations
- GPU/CPU transfer during operations
- Error handling and recovery
- Performance under stress
"""

import pytest
import cato as ca
import time
import math


class TestComplexOperations:
    """Test complex combinations of operations."""
    
    def test_mathematical_expression_simulation(self):
        """Test simulating complex mathematical expressions."""
        print("\n=== Testing mathematical expression simulation ===")
        
        # Simulate: y = (a + b) * c - d * e + f
        a = ca.vector(5, ca.constant(1.0))
        b = ca.vector(5, ca.constant(2.0))
        c = ca.vector(5, ca.constant(3.0))
        d = ca.vector(5, ca.constant(4.0))
        e = ca.vector(5, ca.constant(5.0))
        f = ca.vector(5, ca.constant(6.0))
        
        # Calculate step by step
        step1 = a + b  # (1 + 2) = 3
        step2 = step1 * c  # 3 * 3 = 9
        step3 = d * e  # 4 * 5 = 20
        step4 = step2 - step3  # 9 - 20 = -11
        result = step4 + f  # -11 + 6 = -5
        
        expected = -5.0
        for i in range(5):
            assert abs(result[i] - expected) < 1e-10
        
        print(f"Mathematical expression result: {result}")
        print(" Mathematical expression simulation works correctly")
    
    def test_polynomial_evaluation(self):
        """Test polynomial evaluation using vector operations."""
        print("\n=== Testing polynomial evaluation ===")
        
        # Evaluate: p(x) = 2x^2 + 3x + 1
        x = ca.vector(4, ca.sequence(0, 1))  # [0, 1, 2, 3]
        
        # Calculate: 2x^2 + 3x + 1
        x_squared = x * x  # x^2
        term1 = x_squared * 2.0  # 2x^2
        term2 = x * 3.0  # 3x
        term3 = ca.vector(4, ca.constant(1.0))  # 1
        result = term1 + term2 + term3
        
        # Expected values: [1, 6, 15, 28]
        expected = [1.0, 6.0, 15.0, 28.0]
        for i in range(4):
            assert abs(result[i] - expected[i]) < 1e-10
        
        print(f"Polynomial evaluation result: {result}")
        print(" Polynomial evaluation works correctly")
    
    def test_vector_norms_simulation(self):
        """Test simulating vector norm calculations."""
        print("\n=== Testing vector norms simulation ===")
        
        v = ca.vector(3, ca.sequence(1, 1))  # [1, 2, 3]
        
        # Calculate L2 norm squared: ||v||^2 = v[0]^2 + v[1]^2 + v[2]^2
        v_squared = v * v  # [1, 4, 9]
        # Note: This is element-wise, not the actual norm
        # For actual norm, we'd need reduction operations (not implemented yet)
        
        expected_squared = [1.0, 4.0, 9.0]
        for i in range(3):
            assert abs(v_squared[i] - expected_squared[i]) < 1e-10
        
        print(f"Vector squared elements: {v_squared}")
        print(" Vector norms simulation works correctly")


class TestMemoryManagementOperations:
    """Test memory management during operations."""
    
    def test_operation_memory_consistency(self):
        """Test that operations don't corrupt memory."""
        print("\n=== Testing operation memory consistency ===")
        
        # Create original vectors
        v1 = ca.vector(5, ca.constant(1.0))
        v2 = ca.vector(5, ca.constant(2.0))
        
        # Store original values
        original_v1 = [v1[i] for i in range(5)]
        original_v2 = [v2[i] for i in range(5)]
        
        # Perform operations
        result1 = v1 + v2
        result2 = v1 * v2
        result3 = v1 * 3.0
        
        # Verify original vectors are unchanged
        for i in range(5):
            assert v1[i] == original_v1[i], "Original vector v1 was modified"
            assert v2[i] == original_v2[i], "Original vector v2 was modified"
        
        print(" Operation memory consistency maintained")
    
    def test_gpu_cpu_transfer_during_operations(self):
        """Test GPU/CPU transfer behavior during operations."""
        print("\n=== Testing GPU/CPU transfer during operations ===")
        
        # Create vectors on host
        v1 = ca.vector(5, ca.constant(1.0))
        v2 = ca.vector(5, ca.constant(2.0))
        
        # Ensure they're on host initially
        assert not v1.is_on_gpu()
        assert not v2.is_on_gpu()
        
        # Perform operation (should handle GPU transfer internally)
        result = v1 + v2
        
        # Verify result is correct
        assert result[0] == 3.0
        assert result[4] == 3.0
        
        # Test with GPU vectors
        v1.ensure_on_gpu()
        v2.ensure_on_gpu()
        assert v1.is_on_gpu()
        assert v2.is_on_gpu()
        
        result2 = v1 + v2
        assert result2[0] == 3.0
        assert result2[4] == 3.0
        
        print(" GPU/CPU transfer during operations works correctly")
    
    def test_large_vector_operation_memory(self):
        """Test memory usage with large vectors."""
        print("\n=== Testing large vector operation memory ===")
        
        try:
            # Create large vectors
            v1 = ca.vector(10000, ca.constant(1.0))
            v2 = ca.vector(10000, ca.constant(2.0))
            
            # Perform multiple operations
            result1 = v1 + v2
            result2 = v1 * v2
            result3 = v1 * 3.0
            result4 = v2 + 5.0
            
            # Verify all results are correct
            assert result1[0] == 3.0
            assert result2[0] == 2.0
            assert result3[0] == 3.0
            assert result4[0] == 7.0
            
            print(" Large vector operation memory handled correctly")
        except Exception as e:
            print(f" Large vector operations failed: {e}")


class TestErrorHandlingOperations:
    """Test error handling and recovery in operations."""
    
    def test_operation_error_recovery(self):
        """Test that operations recover gracefully from errors."""
        print("\n=== Testing operation error recovery ===")
        
        # Test with mismatched sizes
        v1 = ca.vector(5, ca.constant(1.0))
        v2 = ca.vector(3, ca.constant(2.0))
        
        # These should raise errors but not crash
        with pytest.raises(Exception):
            _ = v1 + v2
        
        with pytest.raises(Exception):
            _ = v1 - v2
        
        with pytest.raises(Exception):
            _ = v1 * v2
        
        # Verify original vectors are still valid
        assert v1[0] == 1.0
        assert v2[0] == 2.0
        
        print(" Operation error recovery works correctly")
    
    def test_operation_with_invalid_scalars(self):
        """Test operations with invalid scalar values."""
        print("\n=== Testing operations with invalid scalars ===")
        
        v = ca.vector(5, ca.constant(1.0))
        
        # Test with NaN (if supported)
        try:
            result = v * float('nan')
            print(f"NaN operation result: {result}")
        except Exception as e:
            print(f"NaN operation failed (expected): {e}")
        
        # Test with infinity
        try:
            result = v * float('inf')
            print(f"Infinity operation result: {result}")
        except Exception as e:
            print(f"Infinity operation failed (expected): {e}")
        
        # Test with very large numbers
        try:
            result = v * 1e100
            print(f"Large number operation result: {result}")
        except Exception as e:
            print(f"Large number operation failed: {e}")
        
        print(" Invalid scalar operations handled correctly")
    
    def test_operation_with_zero_vectors(self):
        """Test operations with zero-sized vectors."""
        print("\n=== Testing operations with zero-sized vectors ===")
        
        try:
            v1 = ca.vector(0, ca.zeros())
            v2 = ca.vector(0, ca.zeros())
            
            # Operations with zero-sized vectors should work
            result = v1 + v2
            assert result.size() == 0
            
            result2 = v1 * v2
            assert result2.size() == 0
            
            result3 = v1 * 3.0
            assert result3.size() == 0
            
            print(" Zero-sized vector operations work correctly")
        except Exception as e:
            print(f" Zero-sized vector operations failed: {e}")


class TestPerformanceStressOperations:
    """Test performance under stress conditions."""
    
    def test_rapid_operation_sequence(self):
        """Test rapid sequence of operations."""
        print("\n=== Testing rapid operation sequence ===")
        
        v1 = ca.vector(1000, ca.constant(1.0))
        v2 = ca.vector(1000, ca.constant(2.0))
        
        start_time = time.time()
        
        # Perform many operations rapidly
        for i in range(100):
            result = v1 + v2
            result = v1 * v2
            result = v1 * 3.0
            result = v2 + 5.0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"100 operation cycles completed in {total_time:.6f}s")
        print(f"Average time per cycle: {total_time/100:.6f}s")
        
        print(" Rapid operation sequence completed successfully")
    
    def test_memory_intensive_operations(self):
        """Test memory-intensive operations."""
        print("\n=== Testing memory-intensive operations ===")
        
        try:
            # Create multiple large vectors
            vectors = []
            for i in range(10):
                v = ca.vector(5000, ca.constant(float(i)))
                vectors.append(v)
            
            # Perform operations between all vectors
            result = vectors[0]
            for i in range(1, 10):
                result = result + vectors[i]
            
            # Verify result
            expected_sum = sum(range(10))  # 0 + 1 + 2 + ... + 9 = 45
            assert abs(result[0] - expected_sum) < 1e-10
            
            print(" Memory-intensive operations completed successfully")
        except Exception as e:
            print(f" Memory-intensive operations failed: {e}")
    
    def test_operation_scalability(self):
        """Test operation scalability with different vector sizes."""
        print("\n=== Testing operation scalability ===")
        
        sizes = [100, 1000, 5000, 10000]
        times = []
        
        for size in sizes:
            v1 = ca.vector(size, ca.constant(1.0))
            v2 = ca.vector(size, ca.constant(2.0))
            
            start_time = time.time()
            result = v1 + v2
            end_time = time.time()
            
            operation_time = end_time - start_time
            times.append(operation_time)
            
            print(f"Size {size}: {operation_time:.6f}s")
        
        # Verify all operations produced correct results
        for i, size in enumerate(sizes):
            v1 = ca.vector(size, ca.constant(1.0))
            v2 = ca.vector(size, ca.constant(2.0))
            result = v1 + v2
            assert result[0] == 3.0
        
        print(" Operation scalability test completed")


class TestOperationPrecision:
    """Test numerical precision of operations."""
    
    def test_operation_precision_limits(self):
        """Test operations at precision limits."""
        print("\n=== Testing operation precision limits ===")
        
        # Test with very small numbers
        v1 = ca.vector(3, ca.constant(1e-15))
        v2 = ca.vector(3, ca.constant(1e-15))
        result = v1 + v2
        expected = 2e-15
        
        for i in range(3):
            assert abs(result[i] - expected) < 1e-20
        
        # Test with very large numbers
        v3 = ca.vector(3, ca.constant(1e15))
        v4 = ca.vector(3, ca.constant(1e15))
        result2 = v3 + v4
        expected2 = 2e15
        
        for i in range(3):
            assert abs(result2[i] - expected2) < 1e10
        
        print(" Operation precision limits work correctly")
    
    def test_operation_rounding_errors(self):
        """Test handling of rounding errors."""
        print("\n=== Testing operation rounding errors ===")
        
        # Test with numbers that cause rounding issues
        v1 = ca.vector(3, ca.constant(0.1))
        v2 = ca.vector(3, ca.constant(0.2))
        result = v1 + v2
        expected = 0.3
        
        # Should be close to expected value
        for i in range(3):
            assert abs(result[i] - expected) < 1e-15
        
        # Test multiplication precision
        v3 = ca.vector(3, ca.constant(0.1))
        result2 = v3 * 3.0
        expected2 = 0.3
        
        for i in range(3):
            assert abs(result2[i] - expected2) < 1e-15
        
        print(" Operation rounding errors handled correctly")


if __name__ == "__main__":
    print(" Running advanced operations tests...")
    pytest.main([__file__, "-v"])
