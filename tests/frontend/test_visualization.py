#!/usr/bin/env python3
"""
Test for visualization functionality in caVector
"""

import pytest
import cato as ca

def test_smart_string_representation():
    """Test smart string representation like numpy"""
    # Small vector - should show all elements
    V1 = ca.vector(3, ca.zeros())
    str_repr = str(V1)
    assert "caVector([" in str_repr
    assert "0.000000" in str_repr
    assert "size=3" not in str_repr  # Small vectors don't show size
    
    # Large vector - should show first/last elements and size
    V2 = ca.vector(15, ca.ones())
    str_repr = str(V2)
    assert "caVector([" in str_repr
    assert "..." in str_repr
    assert "size=15" in str_repr

def test_to_list_string():
    """Test conversion to list string"""
    V = ca.vector(4, ca.constant(2.5))
    list_str = V.to_list_string()
    assert list_str.startswith("[")
    assert list_str.endswith("]")
    assert "2.500000" in list_str

def test_head_string():
    """Test head string functionality"""
    V = ca.vector(10, ca.arange())
    head_str = V.head_string(3)
    assert head_str.startswith("[")
    assert head_str.endswith("]")
    assert "0.000000" in head_str
    assert "1.000000" in head_str
    assert "2.000000" in head_str

def test_tail_string():
    """Test tail string functionality"""
    V = ca.vector(10, ca.arange())
    tail_str = V.tail_string(3)
    assert tail_str.startswith("[")
    assert tail_str.endswith("]")
    assert "7.000000" in tail_str
    assert "8.000000" in tail_str
    assert "9.000000" in tail_str

def test_visualization_methods_exist():
    """Test that all visualization methods exist"""
    V = ca.vector(5, ca.zeros())
    
    # Check that methods exist
    assert hasattr(V, 'to_list_string')
    assert hasattr(V, 'head_string')
    assert hasattr(V, 'tail_string')
    assert hasattr(V, 'smart_string')

def test_visualization_with_different_sizes():
    """Test visualization with different vector sizes"""
    # Empty-like behavior (edge case)
    V1 = ca.vector(1, ca.zeros())
    assert str(V1) is not None
    
    # Medium vector
    V2 = ca.vector(8, ca.ones())
    assert str(V2) is not None
    
    # Large vector
    V3 = ca.vector(20, ca.random(42))
    assert str(V3) is not None
    assert "..." in str(V3)  # Should show ellipsis

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
