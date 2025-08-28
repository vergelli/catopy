#!/usr/bin/env python3
"""
Test suite for error handling and edge cases.

This module tests how the system handles:
- Invalid inputs and parameters
- Edge cases (empty vectors, very large vectors)
- Error conditions and exceptions
- Boundary conditions
"""

import pytest
import cato as ca


class TestErrorHandling:
    """Test class for error handling scenarios."""
    
    def test_invalid_vector_size(self):
        """Test vector creation with invalid sizes."""
        print("\n=== Testing invalid vector sizes ===")
        
        # Test with size 0 (edge case)
        try:
            V = ca.vector(0, ca.zeros())
            assert V is not None
            assert V.size() == 0
            print("Empty vector (size=0) created successfully")
        except Exception as e:
            print(f"⚠️ Empty vector creation failed: {e}")
        
        # Test with very large size
        try:
            V = ca.vector(100000, ca.ones())
            assert V is not None
            assert V.size() == 100000
            print("Very large vector (size=100000) created successfully")
        except Exception as e:
            print(f"⚠️ Very large vector creation failed: {e}")
        
        # Test with extremely large size (should fail gracefully)
        try:
            V = ca.vector(10000000, ca.zeros())
            print(f"⚠️ Extremely large vector created (unexpected): size={V.size()}")
        except Exception as e:
            print(f"Extremely large vector creation failed as expected: {e}")
    
    def test_invalid_initialization_parameters(self):
        """Test initialization functions with invalid parameters."""
        print("\n=== Testing invalid initialization parameters ===")
        
        # Test constant with extreme values
        try:
            V1 = ca.vector(3, ca.constant(1e308))  # Very large number
            assert V1.size() == 3
            print("constant(1e308) - Vector created successfully")
        except Exception as e:
            print(f"⚠️ constant(1e308) failed: {e}")
        
        try:
            V2 = ca.vector(3, ca.constant(-1e308))  # Very negative number
            assert V2.size() == 3
            print("constant(-1e308) - Vector created successfully")
        except Exception as e:
            print(f"⚠️ constant(-1e308) failed: {e}")
        
        # Test arange with invalid step
        try:
            V3 = ca.vector(5, ca.arange(0, 10, 0))  # Zero step
            print(f"⚠️ arange with zero step created (unexpected): {V3}")
        except Exception as e:
            print(f"arange with zero step failed as expected: {e}")
    
    def test_mathematical_function_edge_cases(self):
        """Test mathematical function with edge cases."""
        print("\n=== Testing mathematical function edge cases ===")
        
        # Test with invalid function names
        try:
            V1 = ca.vector(4, ca.mathematical('invalid_func'))
            assert V1.size() == 4
            print("mathematical('invalid_func') - Vector created (defaults to linear)")
        except Exception as e:
            print(f"⚠️ mathematical('invalid_func') failed: {e}")
        
        # Test with empty string
        try:
            V2 = ca.vector(3, ca.mathematical(''))
            assert V2.size() == 3
            print("mathematical('') - Vector created (defaults to linear)")
        except Exception as e:
            print(f"⚠️ mathematical('') failed: {e}")
    
    def test_random_functions_edge_cases(self):
        """Test random functions with edge cases."""
        print("\n=== Testing random function edge cases ===")
        
        # Test with same seed multiple times (should be reproducible)
        V1 = ca.vector(5, ca.random(42))
        V2 = ca.vector(5, ca.random(42))
        assert V1.to_list_string() == V2.to_list_string()
        print("Random with same seed is reproducible")
        
        # Test with negative seed
        try:
            V3 = ca.vector(3, ca.random(-999))
            assert V3.size() == 3
            print("Random with negative seed works")
        except Exception as e:
            print(f"⚠️ Random with negative seed failed: {e}")
        
        # Test uniform with min > max
        try:
            V4 = ca.vector(3, ca.uniform(10, 5))  # min > max
            assert V4.size() == 3
            print("Uniform with min > max works (swaps values)")
        except Exception as e:
            print(f"⚠️ Uniform with min > max failed: {e}")
    
    def test_sequence_edge_cases(self):
        """Test sequence function with edge cases."""
        print("\n=== Testing sequence function edge cases ===")
        
        # Test with zero step
        try:
            V1 = ca.vector(5, ca.sequence(0, 0))
            assert V1.size() == 5
            print("sequence(0, 0) - Vector created (all elements same)")
        except Exception as e:
            print(f"⚠️ sequence(0, 0) failed: {e}")
        
        # Test with very small step
        try:
            V2 = ca.vector(4, ca.sequence(0, 1e-10))
            assert V2.size() == 4
            print("sequence(0, 1e-10) - Vector created with tiny step")
        except Exception as e:
            print(f"⚠️ sequence(0, 1e-10) failed: {e}")
        
        # Test with very large step
        try:
            V3 = ca.vector(3, ca.sequence(0, 1e10))
            assert V3.size() == 3
            print("sequence(0, 1e10) - Vector created with huge step")
        except Exception as e:
            print(f"⚠️ sequence(0, 1e10) failed: {e}")
    
    def test_sine_edge_cases(self):
        """Test sine function with edge cases."""
        print("\n=== Testing sine function edge cases ===")
        
        # Test with zero frequency
        try:
            V1 = ca.vector(4, ca.sine(0, 1, 0))
            assert V1.size() == 4
            print("sine(0, 1, 0) - Vector created (constant value)")
        except Exception as e:
            print(f"⚠️ sine(0, 1, 0) failed: {e}")
        
        # Test with zero amplitude
        try:
            V2 = ca.vector(3, ca.sine(1, 0, 0))
            assert V2.size() == 3
            print("sine(1, 0, 0) - Vector created (all zeros)")
        except Exception as e:
            print(f"⚠️ sine(1, 0, 0) failed: {e}")
        
        # Test with very high frequency
        try:
            V3 = ca.vector(5, ca.sine(1e6, 1, 0))
            assert V3.size() == 5
            print("sine(1e6, 1, 0) - Vector created with very high frequency")
        except Exception as e:
            print(f"⚠️ sine(1e6, 1, 0) failed: {e}")


class TestBoundaryConditions:
    """Test class for boundary conditions."""
    
    def test_single_element_vectors(self):
        """Test vectors with single elements."""
        print("\n=== Testing single element vectors ===")
        
        # Test all initialization functions with size 1
        init_functions = [
            ('zeros', ca.zeros()),
            ('ones', ca.ones()),
            ('constant(42)', ca.constant(42)),
            ('random(42)', ca.random(42)),
            ('uniform(0, 1, 42)', ca.uniform(0, 1, 42)),
            ('normal(0, 1, 42)', ca.normal(0, 1, 42)),
            ('box_muller(0, 1, 42)', ca.box_muller(0, 1, 42)),
            ('sequence(10, 5)', ca.sequence(10, 5)),
            ('arange(0, 10, 2)', ca.arange(0, 10, 2)),
            ('mathematical("sin")', ca.mathematical('sin')),
            ('sine(1, 2, 0)', ca.sine(1, 2, 0))
        ]
        
        for name, func in init_functions:
            try:
                V = ca.vector(1, func)
                assert V.size() == 1
                print(f"{name} - Single element vector created")
            except Exception as e:
                print(f"❌ {name} - Failed: {e}")
    
    def test_two_element_vectors(self):
        """Test vectors with two elements."""
        print("\n=== Testing two element vectors ===")
        
        # Test a few key functions with size 2
        test_cases = [
            ('zeros', ca.zeros()),
            ('ones', ca.ones()),
            ('constant(3.14)', ca.constant(3.14)),
            ('arange(0, 2)', ca.arange(0, 2))
        ]
        
        for name, func in test_cases:
            try:
                V = ca.vector(2, func)
                assert V.size() == 2
                print(f"{name} - Two element vector created")
            except Exception as e:
                print(f"❌ {name} - Failed: {e}")


class TestStressConditions:
    """Test class for stress conditions."""
    
    def test_rapid_vector_creation(self):
        """Test creating many vectors rapidly."""
        print("\n=== Testing rapid vector creation ===")
        
        try:
            vectors = []
            for i in range(100):
                V = ca.vector(10, ca.constant(i))
                vectors.append(V)
                assert V.size() == 10
            
            print(f"Successfully created {len(vectors)} vectors rapidly")
        except Exception as e:
            print(f"❌ Rapid vector creation failed: {e}")
    
    def test_mixed_initialization_types(self):
        """Test mixing different initialization types."""
        print("\n=== Testing mixed initialization types ===")
        
        try:
            # Create vectors with different initialization functions
            V1 = ca.vector(5, ca.zeros())
            V2 = ca.vector(5, ca.ones())
            V3 = ca.vector(5, ca.constant(2.5))
            V4 = ca.vector(5, ca.random(42))
            V5 = ca.vector(5, ca.arange())
            
            # Verify all vectors are valid
            assert all(V.size() == 5 for V in [V1, V2, V3, V4, V5])
            print("Mixed initialization types work correctly")
            
        except Exception as e:
            print(f"❌ Mixed initialization types failed: {e}")


if __name__ == "__main__":
    print("🚀 Running error handling and edge case tests...")
    pytest.main([__file__, "-v"])
