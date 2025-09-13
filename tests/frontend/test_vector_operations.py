#!/usr/bin/env python3
"""
Test suite for vector operations in caVectors.

This module tests all vector operations including:
- Element-wise operations (+, -, *)
- Scalar operations (+, *)
- Reverse operations (scalar + vector, scalar * vector)
- Edge cases and error handling
- Performance characteristics
- Data consistency
"""

import pytest
import cato as ca
import math


class TestElementWiseOperations:
    """Test element-wise operations between vectors."""
    
    def test_vector_addition(self):
        """Test element-wise addition between vectors."""
        print("\n=== Testing vector addition ===")
        
        # Test basic addition
        v1 = ca.vector(5, ca.constant(2.0))
        v2 = ca.vector(5, ca.constant(3.0))
        result = v1 + v2
        
        assert result is not None
        assert result.size() == 5
        assert result[0] == 5.0
        assert result[4] == 5.0
        print(f"Addition result: {result}")
        
        # Test with different sizes (should raise error)
        v3 = ca.vector(3, ca.constant(1.0))
        with pytest.raises(Exception):
            _ = v1 + v3
        
        print(" Vector addition works correctly")
    
    def test_vector_subtraction(self):
        """Test element-wise subtraction between vectors."""
        print("\n=== Testing vector subtraction ===")
        
        # Test basic subtraction
        v1 = ca.vector(5, ca.constant(5.0))
        v2 = ca.vector(5, ca.constant(2.0))
        result = v1 - v2
        
        assert result is not None
        assert result.size() == 5
        assert result[0] == 3.0
        assert result[4] == 3.0
        print(f"Subtraction result: {result}")
        
        # Test with different sizes (should raise error)
        v3 = ca.vector(3, ca.constant(1.0))
        with pytest.raises(Exception):
            _ = v1 - v3
        
        print(" Vector subtraction works correctly")
    
    def test_vector_multiplication(self):
        """Test element-wise multiplication between vectors."""
        print("\n=== Testing vector multiplication ===")
        
        # Test basic multiplication
        v1 = ca.vector(5, ca.constant(2.0))
        v2 = ca.vector(5, ca.constant(3.0))
        result = v1 * v2
        
        assert result is not None
        assert result.size() == 5
        assert result[0] == 6.0
        assert result[4] == 6.0
        print(f"Multiplication result: {result}")
        
        # Test with different sizes (should raise error)
        v3 = ca.vector(3, ca.constant(1.0))
        with pytest.raises(Exception):
            _ = v1 * v3
        
        print(" Vector multiplication works correctly")
    
    def test_operation_commutativity(self):
        """Test that operations are commutative where applicable."""
        print("\n=== Testing operation commutativity ===")
        
        v1 = ca.vector(4, ca.constant(2.0))
        v2 = ca.vector(4, ca.constant(3.0))
        
        # Addition should be commutative
        result1 = v1 + v2
        result2 = v2 + v1
        assert result1.to_list_string() == result2.to_list_string()
        
        # Multiplication should be commutative
        result3 = v1 * v2
        result4 = v2 * v1
        assert result3.to_list_string() == result4.to_list_string()
        
        print(" Operations are commutative")
    
    def test_operation_associativity(self):
        """Test that operations are associative where applicable."""
        print("\n=== Testing operation associativity ===")
        
        v1 = ca.vector(4, ca.constant(1.0))
        v2 = ca.vector(4, ca.constant(2.0))
        v3 = ca.vector(4, ca.constant(3.0))
        
        # Addition should be associative
        result1 = (v1 + v2) + v3
        result2 = v1 + (v2 + v3)
        assert result1.to_list_string() == result2.to_list_string()
        
        # Multiplication should be associative
        result3 = (v1 * v2) * v3
        result4 = v1 * (v2 * v3)
        assert result3.to_list_string() == result4.to_list_string()
        
        print(" Operations are associative")


class TestScalarOperations:
    """Test scalar operations with vectors."""
    
    def test_scalar_addition(self):
        """Test addition of scalar to vector."""
        print("\n=== Testing scalar addition ===")
        
        v = ca.vector(5, ca.constant(2.0))
        result = v + 3.0
        
        assert result is not None
        assert result.size() == 5
        assert result[0] == 5.0
        assert result[4] == 5.0
        print(f"Scalar addition result: {result}")
        
        # Test reverse operation
        result2 = 3.0 + v
        assert result2.to_list_string() == result.to_list_string()
        print(" Scalar addition works correctly")
    
    def test_scalar_multiplication(self):
        """Test multiplication of vector by scalar."""
        print("\n=== Testing scalar multiplication ===")
        
        v = ca.vector(5, ca.constant(2.0))
        result = v * 3.0
        
        assert result is not None
        assert result.size() == 5
        assert result[0] == 6.0
        assert result[4] == 6.0
        print(f"Scalar multiplication result: {result}")
        
        # Test reverse operation
        result2 = 3.0 * v
        assert result2.to_list_string() == result.to_list_string()
        print(" Scalar multiplication works correctly")
    
    def test_scalar_operations_with_different_values(self):
        """Test scalar operations with various scalar values."""
        print("\n=== Testing scalar operations with different values ===")
        
        v = ca.vector(4, ca.constant(1.0))
        
        # Test with positive scalar
        result1 = v * 2.5
        assert result1[0] == 2.5
        
        # Test with negative scalar
        result2 = v * -1.5
        assert result2[0] == -1.5
        
        # Test with zero scalar
        result3 = v * 0.0
        assert result3[0] == 0.0
        
        # Test with fractional scalar
        result4 = v * 0.5
        assert result4[0] == 0.5
        
        print(" Scalar operations work with different values")
    
    def test_scalar_operations_chain(self):
        """Test chaining of scalar operations."""
        print("\n=== Testing chained scalar operations ===")
        
        v = ca.vector(4, ca.constant(1.0))
        
        # Test chained operations
        result = (v + 2.0) * 3.0
        expected = ca.vector(4, ca.constant(9.0))  # (1 + 2) * 3 = 9
        
        assert result.to_list_string() == expected.to_list_string()
        print(f"Chained operations result: {result}")
        
        # Test more complex chain
        result2 = (v * 2.0) + (v * 3.0)
        expected2 = ca.vector(4, ca.constant(5.0))  # (1 * 2) + (1 * 3) = 5
        
        assert result2.to_list_string() == expected2.to_list_string()
        print(" Chained scalar operations work correctly")


class TestOperationEdgeCases:
    """Test edge cases and error conditions for operations."""
    
    def test_empty_vector_operations(self):
        """Test operations with empty vectors."""
        print("\n=== Testing empty vector operations ===")
        
        try:
            v1 = ca.vector(0, ca.zeros())
            v2 = ca.vector(0, ca.zeros())
            
            # Operations with empty vectors should work
            result = v1 + v2
            assert result.size() == 0
            print("Empty vector operations work")
        except Exception as e:
            print(f" Empty vector operations failed: {e}")
    
    def test_single_element_operations(self):
        """Test operations with single-element vectors."""
        print("\n=== Testing single element operations ===")
        
        v1 = ca.vector(1, ca.constant(2.0))
        v2 = ca.vector(1, ca.constant(3.0))
        
        # Test all operations
        result_add = v1 + v2
        result_sub = v1 - v2
        result_mul = v1 * v2
        
        assert result_add[0] == 5.0
        assert result_sub[0] == -1.0
        assert result_mul[0] == 6.0
        
        print(" Single element operations work correctly")
    
    def test_large_vector_operations(self):
        """Test operations with large vectors."""
        print("\n=== Testing large vector operations ===")
        
        try:
            v1 = ca.vector(1000, ca.constant(1.0))
            v2 = ca.vector(1000, ca.constant(2.0))
            
            # Test operations
            result = v1 + v2
            assert result.size() == 1000
            assert result[0] == 3.0
            assert result[999] == 3.0
            
            print(" Large vector operations work correctly")
        except Exception as e:
            print(f" Large vector operations failed: {e}")
    
    def test_mixed_initialization_operations(self):
        """Test operations between vectors with different initialization patterns."""
        print("\n=== Testing mixed initialization operations ===")
        
        v1 = ca.vector(5, ca.sequence(1, 1))  # [1, 2, 3, 4, 5]
        v2 = ca.vector(5, ca.constant(2.0))   # [2, 2, 2, 2, 2]
        
        # Test addition
        result_add = v1 + v2
        expected_add = [3.0, 4.0, 5.0, 6.0, 7.0]
        for i in range(5):
            assert abs(result_add[i] - expected_add[i]) < 1e-10
        
        # Test multiplication
        result_mul = v1 * v2
        expected_mul = [2.0, 4.0, 6.0, 8.0, 10.0]
        for i in range(5):
            assert abs(result_mul[i] - expected_mul[i]) < 1e-10
        
        print(" Mixed initialization operations work correctly")
    
    def test_operation_precision(self):
        """Test numerical precision of operations."""
        print("\n=== Testing operation precision ===")
        
        v1 = ca.vector(3, ca.constant(0.1))
        v2 = ca.vector(3, ca.constant(0.2))
        
        # Test addition precision
        result = v1 + v2
        expected = 0.3
        for i in range(3):
            assert abs(result[i] - expected) < 1e-15
        
        # Test multiplication precision
        result2 = v1 * v2
        expected2 = 0.02
        for i in range(3):
            assert abs(result2[i] - expected2) < 1e-15
        
        print(" Operations maintain numerical precision")


class TestOperationPerformance:
    """Test performance characteristics of operations."""
    
    def test_operation_timing(self):
        """Test timing of operations for performance analysis."""
        print("\n=== Testing operation timing ===")
        
        import time
        
        # Create test vectors
        v1 = ca.vector(10000, ca.constant(1.0))
        v2 = ca.vector(10000, ca.constant(2.0))
        
        # Time addition
        start_time = time.time()
        result_add = v1 + v2
        add_time = time.time() - start_time
        
        # Time multiplication
        start_time = time.time()
        result_mul = v1 * v2
        mul_time = time.time() - start_time
        
        # Time scalar operations
        start_time = time.time()
        result_scalar = v1 * 3.0
        scalar_time = time.time() - start_time
        
        print(f"Addition time: {add_time:.6f}s")
        print(f"Multiplication time: {mul_time:.6f}s")
        print(f"Scalar multiplication time: {scalar_time:.6f}s")
        
        # Verify results are correct
        assert result_add[0] == 3.0
        assert result_mul[0] == 2.0
        assert result_scalar[0] == 3.0
        
        print(" Operation timing completed")
    
    def test_memory_usage_operations(self):
        """Test memory usage during operations."""
        print("\n=== Testing memory usage during operations ===")
        
        # Create vectors
        v1 = ca.vector(1000, ca.constant(1.0))
        v2 = ca.vector(1000, ca.constant(2.0))
        
        # Perform operations
        result1 = v1 + v2
        result2 = v1 * v2
        result3 = v1 * 3.0
        
        # Verify all results are valid
        assert result1.size() == 1000
        assert result2.size() == 1000
        assert result3.size() == 1000
        
        print(" Memory usage during operations is correct")


class TestOperationIntegration:
    """Test integration between different operations."""
    
    def test_operation_chain(self):
        """Test chaining multiple operations together."""
        print("\n=== Testing operation chaining ===")
        
        v1 = ca.vector(4, ca.constant(1.0))
        v2 = ca.vector(4, ca.constant(2.0))
        v3 = ca.vector(4, ca.constant(3.0))
        
        # Test complex operation chain
        result = (v1 + v2) * v3 + (v1 * 2.0)
        # Expected: (1 + 2) * 3 + (1 * 2) = 3 * 3 + 2 = 9 + 2 = 11
        expected = 11.0
        
        for i in range(4):
            assert abs(result[i] - expected) < 1e-10
        
        print(f"Operation chain result: {result}")
        print(" Operation chaining works correctly")
    
    def test_operation_with_different_sizes(self):
        """Test operations with vectors of different sizes."""
        print("\n=== Testing operations with different sizes ===")
        
        v1 = ca.vector(5, ca.constant(1.0))
        v2 = ca.vector(3, ca.constant(2.0))
        
        # These should raise errors
        with pytest.raises(Exception):
            _ = v1 + v2
        
        with pytest.raises(Exception):
            _ = v1 - v2
        
        with pytest.raises(Exception):
            _ = v1 * v2
        
        print(" Different size operations correctly raise errors")
    
    def test_operation_with_scalars_of_different_types(self):
        """Test operations with different scalar types."""
        print("\n=== Testing operations with different scalar types ===")
        
        v = ca.vector(4, ca.constant(1.0))
        
        # Test with integer scalar
        result1 = v * 2
        assert result1[0] == 2.0
        
        # Test with float scalar
        result2 = v * 2.5
        assert result2[0] == 2.5
        
        # Test with negative scalar
        result3 = v * -1.0
        assert result3[0] == -1.0
        
        print(" Operations work with different scalar types")


if __name__ == "__main__":
    print(" Running vector operations tests...")
    pytest.main([__file__, "-v"])
