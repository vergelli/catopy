#!/usr/bin/env python3
"""
Basic functionality test for frontend (Python bindings).

This module tests the core functionality of caVector objects:
- Import and module availability
- Basic vector creation and properties
- Memory management and lazy copy
- Basic vector operations and methods
"""

import pytest
import cato as ca


class TestBasicFunctionality:
    """Test class for basic vector functionality."""
    
    def test_import(self):
        """Test that cato module can be imported successfully."""
        assert ca is not None
        print("Cato module imported successfully")
    
    def test_module_attributes(self):
        """Test that all expected module attributes exist."""
        print("\n=== Testing module attributes ===")

        # Core classes
        assert hasattr(ca, 'caVector'), "caVeector class not found"
        assert hasattr(ca, 'Devices'), "Devices class not found"
        print("Core classes found")
        
        # Convenience functions
        assert hasattr(ca, 'vector'), "vector() function not found"
        print("Convenience functions found")
        
        # Initialization functions (just check they exist)
        init_functions = ['zeros', 'ones', 'constant', 'random', 'uniform', 
                         'normal', 'box_muller', 'sequence', 'arange', 
                         'mathematical', 'sine']
        
        for func_name in init_functions:
            assert hasattr(ca, func_name), f"Initialization function {func_name} not found"
        print("All initialization functions found")
    
    def test_basic_vector_creation(self):
        """Test basic vector creation with different initialization functions."""
        print("\n=== Testing basic vector creation ===")
        
        # Test with zeros
        V1 = ca.vector(5, ca.zeros())
        assert V1 is not None
        assert V1.size() == 5
        print(f"Vector with zeros: {V1}")
        
        # Test with ones
        V2 = ca.vector(3, ca.ones())
        assert V2 is not None
        assert V2.size() == 3
        print(f"Vector with ones: {V2}")
        
        # Test with constant
        V3 = ca.vector(4, ca.constant(2.5))
        assert V3 is not None
        assert V3.size() == 4
        print(f"Vector with constant: {V3}")
    
    def test_vector_properties(self):
        """Test basic vector properties and methods."""
        print("\n=== Testing vector properties ===")
        
        V = ca.vector(6, ca.arange())
        assert V is not None
        
        # Test size property
        assert V.size() == 6
        print(f"Vector size: {V.size()}")
        
        # Test string representation
        str_repr = str(V)
        assert str_repr is not None
        print(f"String representation: {str_repr[:50]}...")

        # Test list string conversion
        list_str = V.to_list_string()
        assert list_str is not None
        assert list_str.startswith("[")
        assert list_str.endswith("]")
        print(f"List string: {list_str}")
    
    def test_vector_visualization_methods(self):
        """Test vector visualization and display methods."""
        print("\n=== Testing visualization methods ===")
        
        V = ca.vector(10, ca.sequence(0, 0.5))
        assert V is not None
        
        # Test head_string
        head_str = V.head_string(3)
        assert head_str is not None
        assert head_str.startswith("[")
        assert head_str.endswith("]")
        print(f"Head string (3): {head_str}")
        
        # Test tail_string
        tail_str = V.tail_string(3)
        assert tail_str is not None
        assert tail_str.startswith("[")
        assert tail_str.endswith("]")
        print(f"Tail string (3): {tail_str}")
        
        # Test smart_string
        smart_str = V.smart_string()
        assert smart_str is not None
        print(f"Smart string: {smart_str[:50]}...")
    
    def test_memory_management(self):
        """Test memory management and lazy copy functionality."""
        print("\n=== Testing memory management ===")
        
        V = ca.vector(4, ca.ones())
        assert V is not None
        
        # Initially should be on host
        assert not V.is_on_gpu()
        print("Vector initially on host")
        
        # Test GPU transfer
        V.ensure_on_gpu()
        assert V.is_on_gpu()
        print("Vector transferred to GPU")
        
        # Test host transfer (bug has been fixed!)
        V.ensure_on_host()
        assert not V.is_on_gpu()  # This should now work correctly
        print("Vector transferred back to host")
    
    def test_different_vector_sizes(self):
        """Test vector creation with different sizes."""
        print("\n=== Testing different vector sizes ===")
        
        # Small vector
        V1 = ca.vector(1, ca.constant(42.0))
        assert V1.size() == 1
        print(f"Small vector (size=1): {V1}")
        
        # Medium vector
        V2 = ca.vector(10, ca.zeros())
        assert V2.size() == 10
        print(f"Medium vector (size=10): {V2}")
        
        # Large vector
        V3 = ca.vector(100, ca.ones())
        assert V3.size() == 100
        print(f"Large vector (size=100): {V3}")
    
    def test_vector_consistency(self):
        """Test that vector data is consistent across operations."""
        print("\n=== Testing vector consistency ===")
        
        # Create vector with known pattern
        V = ca.vector(5, ca.sequence(1, 2))
        assert V.size() == 5
        
        # Get string representation
        str_repr = str(V)
        list_str = V.to_list_string()
        
        # Both should contain the same data
        assert "1.000000" in str_repr
        assert "3.000000" in str_repr
        assert "5.000000" in str_repr
        assert "7.000000" in str_repr
        assert "9.000000" in str_repr
        
        print(f"Vector data consistent: {list_str}")
    
    def test_error_handling_basics(self):
        """Test basic error handling for invalid inputs."""
        print("\n=== Testing basic error handling ===")
        
        # Test with invalid size (should handle gracefully)
        try:
            V = ca.vector(0, ca.zeros())
            print(f"Empty vector created: {V}")
        except Exception as e:
            print(f" Empty vector creation failed (expected): {e}")
        
        # Test with very large size (should handle gracefully)
        try:
            V = ca.vector(10000, ca.ones())
            print(f"Large vector created: size={V.size()}")
        except Exception as e:
            print(f" Large vector creation failed: {e}")


if __name__ == "__main__":
    print(" Running basic functionality tests...")
    pytest.main([__file__, "-v"])
