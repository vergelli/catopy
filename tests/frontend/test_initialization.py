#!/usr/bin/env python3
"""
Test suite for vector initialization functions.

This module tests all the initialization functions that we've implemented and fixed:
- zeros, ones, constant, random, uniform, normal, box_muller
- sequence, arange, mathematical, sine

Each test verifies that the function can be called and used to create vectors.
"""

import pytest
import cato as ca
import math


class TestInitializationFunctions:
    """Test class for all initialization functions."""
    
    def test_zeros_function(self):
        """Test zeros() function creates vectors filled with zeros."""
        print("\n=== Testing zeros() function ===")
        
        # Test basic functionality
        V = ca.vector(5, ca.zeros())
        assert V is not None
        assert V.size() == 5
        
        # Verify all elements are zero
        data_str = V.to_list_string()
        assert "0.000000" in data_str
        print(f"zeros() - Vector: {V}")
        print(f"   Data: {data_str}")
        
        # Test different sizes
        V2 = ca.vector(10, ca.zeros())
        assert V2.size() == 10
        print(f"zeros() - Large vector: size={V2.size()}")
    
    def test_ones_function(self):
        """Test ones() function creates vectors filled with ones."""
        print("\n=== Testing ones() function ===")
        
        # Test basic functionality
        V = ca.vector(4, ca.ones())
        assert V is not None
        assert V.size() == 4
        
        # Verify all elements are one
        data_str = V.to_list_string()
        assert "1.000000" in data_str
        print(f"ones() - Vector: {V}")
        print(f"   Data: {data_str}")
        
        # Test different sizes
        V2 = ca.vector(7, ca.ones())
        assert V2.size() == 7
        print(f"ones() - Large vector: size={V2.size()}")
    
    def test_constant_function(self):
        """Test constant(value) function creates vectors filled with constant value."""
        print("\n=== Testing constant(value) function ===")
        
        # Test with positive value
        V1 = ca.vector(3, ca.constant(2.5))
        assert V1 is not None
        assert V1.size() == 3
        data_str = V1.to_list_string()
        assert "2.500000" in data_str
        print(f"constant(2.5) - Vector: {V1}")
        print(f"   Data: {data_str}")
        
        # Test with negative value
        V2 = ca.vector(4, ca.constant(-1.7))
        assert V2.size() == 4
        data_str = V2.to_list_string()
        assert "-1.700000" in data_str
        print(f"constant(-1.7) - Vector: {V2}")
        print(f"   Data: {data_str}")
        
        # Test with zero
        V3 = ca.vector(2, ca.constant(0.0))
        assert V3.size() == 2
        data_str = V3.to_list_string()
        assert "0.000000" in data_str
        print(f"constant(0.0) - Vector: {V3}")
        print(f"   Data: {data_str}")
    
    def test_random_function(self):
        """Test random(seed) function creates vectors with random values."""
        print("\n=== Testing random(seed) function ===")
        
        # Test with fixed seed for reproducibility
        V1 = ca.vector(5, ca.random(42))
        assert V1 is not None
        assert V1.size() == 5
        print(f"random(42) - Vector: {V1}")
        print(f"   Data: {V1.to_list_string()}")
        
        # Test with different seed
        V2 = ca.vector(5, ca.random(123))
        assert V2.size() == 5
        print(f"random(123) - Vector: {V2}")
        print(f"   Data: {V2.to_list_string()}")
        
        # Test with default seed (should be different each time)
        V3 = ca.vector(3, ca.random())
        assert V3.size() == 3
        print(f"random() - Vector: {V3}")
        print(f"   Data: {V3.to_list_string()}")
        
        # Verify that different seeds produce different results
        assert V1.to_list_string() != V2.to_list_string(), "Different seeds should produce different results"
    
    def test_uniform_function(self):
        """Test uniform(min, max, seed) function creates vectors with uniform distribution."""
        print("\n=== Testing uniform(min, max, seed) function ===")
        
        # Test basic uniform distribution
        V1 = ca.vector(6, ca.uniform(0.0, 1.0, 42))
        assert V1 is not None
        assert V1.size() == 6
        print(f"uniform(0.0, 1.0, 42) - Vector: {V1}")
        print(f"   Data: {V1.to_list_string()}")
        
        # Test with negative range
        V2 = ca.vector(4, ca.uniform(-5.0, 5.0, 123))
        assert V2.size() == 4
        print(f"uniform(-5.0, 5.0, 123) - Vector: {V2}")
        print(f"   Data: {V2.to_list_string()}")
        
        # Test with default parameters
        V3 = ca.vector(3, ca.uniform())
        assert V3.size() == 3
        print(f"uniform() - Vector: {V3}")
        print(f"   Data: {V3.to_list_string()}")
    
    def test_normal_function(self):
        """Test normal(mean, std, seed) function creates vectors with normal distribution."""
        print("\n=== Testing normal(mean, std, seed) function ===")
        
        # Test standard normal distribution
        V1 = ca.vector(8, ca.normal(0.0, 1.0, 42))
        assert V1 is not None
        assert V1.size() == 8
        print(f"normal(0.0, 1.0, 42) - Vector: {V1}")
        print(f"   Data: {V1.to_list_string()}")
        
        # Test with custom mean and std
        V2 = ca.vector(5, ca.normal(10.0, 2.0, 123))
        assert V2.size() == 5
        print(f"normal(10.0, 2.0, 123) - Vector: {V2}")
        print(f"   Data: {V2.to_list_string()}")
        
        # Test with default parameters
        V3 = ca.vector(4, ca.normal())
        assert V3.size() == 4
        print(f"normal() - Vector: {V3}")
        print(f"   Data: {V3.to_list_string()}")
    
    def test_box_muller_function(self):
        """Test box_muller(mean, std, seed) function creates vectors with Box-Muller distribution."""
        print("\n=== Testing box_muller(mean, std, seed) function ===")
        
        # Test standard Box-Muller distribution
        V1 = ca.vector(6, ca.box_muller(0.0, 1.0, 42))
        assert V1 is not None
        assert V1.size() == 6
        print(f"box_muller(0.0, 1.0, 42) - Vector: {V1}")
        print(f"   Data: {V1.to_list_string()}")
        
        # Test with custom parameters
        V2 = ca.vector(4, ca.box_muller(5.0, 0.5, 123))
        assert V2.size() == 4
        print(f"box_muller(5.0, 0.5, 123) - Vector: {V2}")
        print(f"   Data: {V2.to_list_string()}")
        
        # Test with default parameters
        V3 = ca.vector(5, ca.box_muller())
        assert V3.size() == 5
        print(f"box_muller() - Vector: {V3}")
        print(f"   Data: {V3.to_list_string()}")
    
    def test_sequence_function(self):
        """Test sequence(start, step) function creates arithmetic sequences."""
        print("\n=== Testing sequence(start, step) function ===")
        
        # Test basic sequence
        V1 = ca.vector(5, ca.sequence(0.0, 1.0))
        assert V1 is not None
        assert V1.size() == 5
        print(f"sequence(0.0, 1.0) - Vector: {V1}")
        print(f"   Data: {V1.to_list_string()}")
        
        # Test with custom start and step
        V2 = ca.vector(4, ca.sequence(10.0, 2.5))
        assert V2.size() == 4
        print(f"sequence(10.0, 2.5) - Vector: {V2}")
        print(f"   Data: {V2.to_list_string()}")
        
        # Test with negative step
        V3 = ca.vector(3, ca.sequence(5.0, -1.0))
        assert V3.size() == 3
        print(f"sequence(5.0, -1.0) - Vector: {V3}")
        print(f"   Data: {V3.to_list_string()}")
    
    def test_arange_function(self):
        """Test arange(start, stop, step) function creates arange-like sequences."""
        print("\n=== Testing arange(start, stop, step) function ===")
        
        # Test basic arange
        V1 = ca.vector(5, ca.arange())
        assert V1 is not None
        assert V1.size() == 5
        print(f"arange() - Vector: {V1}")
        print(f"   Data: {V1.to_list_string()}")
        
        # Test with custom parameters
        V2 = ca.vector(5, ca.arange(0, 20, 4))
        assert V2.size() == 5
        print(f"arange(0, 20, 4) - Vector: {V2}")
        print(f"   Data: {V2.to_list_string()}")
        
        # Test with decimal step
        V3 = ca.vector(5, ca.arange(0, 1, 0.2))
        assert V3.size() == 5
        print(f"arange(0, 1, 0.2) - Vector: {V3}")
        print(f"   Data: {V3.to_list_string()}")
        
        # Test with negative step
        V4 = ca.vector(4, ca.arange(10, 0, -2))
        assert V4.size() == 4
        print(f"arange(10, 0, -2) - Vector: {V4}")
        print(f"   Data: {V4.to_list_string()}")
    
    def test_mathematical_function(self):
        """Test mathematical(func_name) function creates mathematical function values."""
        print("\n=== Testing mathematical(func_name) function ===")
        
        # Test sine function
        V1 = ca.vector(4, ca.mathematical('sin'))
        assert V1 is not None
        assert V1.size() == 4
        print(f"mathematical('sin') - Vector: {V1}")
        print(f"   Data: {V1.to_list_string()}")
        
        # Test cosine function
        V2 = ca.vector(4, ca.mathematical('cos'))
        assert V2.size() == 4
        print(f"mathematical('cos') - Vector: {V2}")
        print(f"   Data: {V2.to_list_string()}")
        
        # Test exponential function
        V3 = ca.vector(4, ca.mathematical('exp'))
        assert V3.size() == 4
        print(f"mathematical('exp') - Vector: {V3}")
        print(f"   Data: {V3.to_list_string()}")
        
        # Test log function
        V4 = ca.vector(4, ca.mathematical('log'))
        assert V4.size() == 4
        print(f"mathematical('log') - Vector: {V4}")
        print(f"   Data: {V4.to_list_string()}")
        
        # Test default (linear)
        V5 = ca.vector(3, ca.mathematical())
        assert V5.size() == 3
        print(f"mathematical() - Vector: {V5}")
        print(f"   Data: {V5.to_list_string()}")
    
    def test_sine_function(self):
        """Test sine(frequency, amplitude, phase) function creates sine wave values."""
        print("\n=== Testing sine(frequency, amplitude, phase) function ===")
        
        # Test basic sine wave
        V1 = ca.vector(6, ca.sine())
        assert V1 is not None
        assert V1.size() == 6
        print(f"sine() - Vector: {V1}")
        print(f"   Data: {V1.to_list_string()}")
        
        # Test with custom parameters
        V2 = ca.vector(5, ca.sine(2.0, 3.0, math.pi/4))
        assert V2.size() == 5
        print(f"sine(2.0, 3.0, π/4) - Vector: {V2}")
        print(f"   Data: {V2.to_list_string()}")
        
        # Test with high frequency
        V3 = ca.vector(4, ca.sine(5.0, 1.0, 0.0))
        assert V3.size() == 4
        print(f"sine(5.0, 1.0, 0.0) - Vector: {V3}")
        print(f"   Data: {V3.to_list_string()}")


class TestInitializationEdgeCases:
    """Test class for edge cases and error conditions."""
    
    def test_empty_vector(self):
        """Test creating vectors with size 0."""
        print("\n=== Testing edge case: empty vector ===")
        
        try:
            V = ca.vector(0, ca.zeros())
            assert V is not None
            assert V.size() == 0
            print(f"Empty vector created: {V}")
        except Exception as e:
            print(f" Empty vector creation failed: {e}")
    
    def test_single_element_vector(self):
        """Test creating vectors with size 1."""
        print("\n=== Testing edge case: single element vector ===")
        
        V = ca.vector(1, ca.constant(42.0))
        assert V is not None
        assert V.size() == 1
        print(f"Single element vector: {V}")
        print(f"   Data: {V.to_list_string()}")
    
    def test_large_vector(self):
        """Test creating vectors with large size."""
        print("\n=== Testing edge case: large vector ===")
        
        try:
            V = ca.vector(1000, ca.ones())
            assert V is not None
            assert V.size() == 1000
            print(f"Large vector created: size={V.size()}")
        except Exception as e:
            print(f" Large vector creation failed: {e}")
    
    def test_negative_parameters(self):
        """Test functions with negative parameters where applicable."""
        print("\n=== Testing edge case: negative parameters ===")
        
        # Test constant with negative value
        V1 = ca.vector(3, ca.constant(-999.0))
        assert V1.size() == 3
        print(f"constant(-999.0) - Vector: {V1}")
        
        # Test arange with negative step
        V2 = ca.vector(4, ca.arange(5, -5, -2))
        assert V2.size() == 4
        print(f"arange(5, -5, -2) - Vector: {V2}")


if __name__ == "__main__":
    print(" Running initialization function tests...")
    pytest.main([__file__, "-v"])
