"""
Test suite for indexing operators in caVectors.

This suite tests:
- __getitem__ operator (v[i])
- __setitem__ operator (v[i] = value)
- at() method with bounds checking
- Bounds checking and error handling
- Data consistency after modifications
- Integration with Python iteration protocols
"""

import pytest
import cato as ca


class TestIndexingOperators:
    """Test indexing operators and related functionality."""
    
    def test_basic_getitem(self):
        """Test basic getitem functionality."""
        print("\n=== Testing basic getitem ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # Test basic access
        assert v[0] == 3.1415, "First element should be accessible"
        assert v[4] == 3.1415, "Last element should be accessible"
        assert v[2] == 3.1415, "Middle element should be accessible"
        
        print(" Basic getitem works correctly")
    
    def test_basic_setitem(self):
        """Test basic setitem functionality."""
        print("\n=== Testing basic setitem ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # Test basic assignment
        v[0] = 999.0
        v[2] = 888.0
        v[4] = 777.0
        
        # Verify assignments
        assert v[0] == 999.0, "First element should be modified"
        assert v[2] == 888.0, "Middle element should be modified"
        assert v[4] == 777.0, "Last element should be modified"
        assert v[1] == 3.1415, "Unmodified element should remain unchanged"
        assert v[3] == 3.1415, "Unmodified element should remain unchanged"
        
        print(" Basic setitem works correctly")
    
    def test_bounds_checking_getitem(self):
        """Test bounds checking in getitem."""
        print("\n=== Testing bounds checking in getitem ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # Valid indices should work
        v[0]   # Should not raise
        v[4]   # Should not raise
        
        # Invalid indices should raise IndexError
        with pytest.raises(IndexError):
            v[5]  # Out of bounds
        
        with pytest.raises(IndexError):
            v[100]  # Way out of bounds
        
        print(" Bounds checking works correctly in getitem")
    
    def test_bounds_checking_setitem(self):
        """Test bounds checking in setitem."""
        print("\n=== Testing bounds checking in setitem ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # Valid indices should work
        v[0] = 999.0   # Should not raise
        v[4] = 777.0   # Should not raise
        
        # Invalid indices should raise IndexError
        with pytest.raises(IndexError):
            v[5] = 999.0  # Out of bounds
        
        with pytest.raises(IndexError):
            v[100] = 999.0  # Way out of bounds
        
        print(" Bounds checking works correctly in setitem")
    
    def test_at_method_bounds_checking(self):
        """Test at() method with bounds checking."""
        print("\n=== Testing at() method bounds checking ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # Valid indices should work
        assert v.at(0) == 3.1415, "at(0) should work"
        assert v.at(4) == 3.1415, "at(4) should work"
        
        # Invalid indices should raise IndexError with descriptive message
        with pytest.raises(IndexError) as exc_info:
            v.at(5)
        assert "index 5 out of range" in str(exc_info.value), "Error message should be descriptive"
        
        print(" at() method bounds checking works correctly")
    
    def test_at_method_modification(self):
        """Test at() method for modification."""
        print("\n=== Testing at() method for modification ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # Modify using regular indexing (which uses __setitem__)
        v[0] = 999.0
        v[2] = 888.0
        v[4] = 777.0
        
        # Verify modifications using at() for reading
        assert v.at(0) == 999.0, "at(0) should read modified value"
        assert v.at(2) == 888.0, "at(2) should read modified value"
        assert v.at(4) == 777.0, "at(4) should read modified value"
        
        # Verify using regular indexing
        assert v[0] == 999.0, "Regular indexing should reflect modifications"
        assert v[2] == 888.0, "Regular indexing should reflect modifications"
        assert v[4] == 777.0, "Regular indexing should reflect modifications"
        
        print(" at() method modification works correctly")
    
    def test_data_consistency_after_modifications(self):
        """Test that data remains consistent after modifications."""
        print("\n=== Testing data consistency after modifications ===")
        
        v = ca.vector(10, ca.constant(1.0))
        
        # Modify multiple elements
        modifications = {
            0: 100.0,
            3: 300.0,
            7: 700.0,
            9: 900.0
        }
        
        for index, value in modifications.items():
            v[index] = value
        
        # Verify all modifications
        for index, expected_value in modifications.items():
            assert v[index] == expected_value, f"Element {index} should be {expected_value}"
        
        # Verify unmodified elements
        for i in range(10):
            if i not in modifications:
                assert v[i] == 1.0, f"Unmodified element {i} should remain 1.0"
        
        print(" Data consistency maintained after modifications")
    
    def test_python_iteration_protocol(self):
        """Test that caVector works with Python iteration protocols."""
        print("\n=== Testing Python iteration protocols ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # Test len()
        assert len(v) == 5, "len() should work correctly"
        
        # Test iteration
        values = list(v)
        assert len(values) == 5, "Iteration should produce correct number of elements"
        assert all(x == 3.1415 for x in values), "All elements should be 3.1415"
        
        # Test list comprehension
        doubled = [x * 2 for x in v]
        assert doubled == [6.283, 6.283, 6.283, 6.283, 6.283], "List comprehension should work"
        
        print(" Python iteration protocols work correctly")
    
    def test_conversion_to_python_types(self):
        """Test conversion to Python types."""
        print("\n=== Testing conversion to Python types ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # Test conversion to list
        v_list = list(v)
        assert isinstance(v_list, list), "Should convert to list"
        assert v_list == [3.1415, 3.1415, 3.1415, 3.1415, 3.1415], "List should contain correct values"
        
        # Test conversion to tuple
        v_tuple = tuple(v)
        assert isinstance(v_tuple, tuple), "Should convert to tuple"
        assert v_tuple == (3.1415, 3.1415, 3.1415, 3.1415, 3.1415), "Tuple should contain correct values"
        
        # Test conversion to set
        v_set = set(v)
        assert isinstance(v_set, set), "Should convert to set"
        assert v_set == {3.1415}, "Set should contain unique values"
        
        print(" Conversion to Python types works correctly")
    
    def test_modification_and_iteration(self):
        """Test that modifications work correctly with iteration."""
        print("\n=== Testing modification and iteration ===")
        
        v = ca.vector(5, ca.constant(1.0))
        
        # Modify elements
        v[0] = 10.0
        v[2] = 30.0
        v[4] = 50.0
        
        # Convert to list
        v_list = list(v)
        expected = [10.0, 1.0, 30.0, 1.0, 50.0]
        assert v_list == expected, "List should reflect modifications"
        
        # Test list comprehension with modified values
        squared = [x * x for x in v]
        expected_squared = [100.0, 1.0, 900.0, 1.0, 2500.0]
        assert squared == expected_squared, "List comprehension should work with modified values"
        
        print(" Modification and iteration work together correctly")
    
    def test_edge_cases_indexing(self):
        """Test edge cases in indexing."""
        print("\n=== Testing edge cases in indexing ===")
        
        # Single element vector
        v1 = ca.vector(1, ca.constant(42.0))
        assert v1[0] == 42.0, "Single element should be accessible"
        v1[0] = 999.0
        assert v1[0] == 999.0, "Single element should be modifiable"
        
        # Two element vector
        v2 = ca.vector(2, ca.constant(3.14))
        assert v2[0] == 3.14, "First element should be accessible"
        assert v2[1] == 3.14, "Second element should be accessible"
        v2[0] = 100.0
        v2[1] = 200.0
        assert v2[0] == 100.0, "First element should be modifiable"
        assert v2[1] == 200.0, "Second element should be modifiable"
        
        print(" Edge cases in indexing work correctly")
    
    def test_large_vector_indexing(self):
        """Test indexing with larger vectors."""
        print("\n=== Testing large vector indexing ===")
        
        # Create larger vector
        v = ca.vector(1000, ca.constant(1.0))
        
        # Test access at different positions
        assert v[0] == 1.0, "First element should be accessible"
        assert v[500] == 1.0, "Middle element should be accessible"
        assert v[999] == 1.0, "Last element should be accessible"
        
        # Test modification at different positions
        v[0] = 999.0
        v[500] = 888.0
        v[999] = 777.0
        
        # Verify modifications
        assert v[0] == 999.0, "First element modification should work"
        assert v[500] == 888.0, "Middle element modification should work"
        assert v[999] == 777.0, "Last element modification should work"
        
        print(" Large vector indexing works correctly")
    
    def test_different_initialization_patterns(self):
        """Test indexing with different initialization patterns."""
        print("\n=== Testing indexing with different initialization patterns ===")
        
        # Test with sequence
        v1 = ca.vector(5, ca.sequence(1, 2))
        assert v1[0] == 1.0, "Sequence first element should be 1.0"
        assert v1[1] == 3.0, "Sequence second element should be 3.0"
        assert v1[4] == 9.0, "Sequence last element should be 9.0"
        
        # Test with random
        v2 = ca.vector(5, ca.random(42))
        # Random values should be accessible (we can't predict exact values)
        assert 0.0 <= v2[0] <= 1.0, "Random value should be in [0,1] range"
        assert 0.0 <= v2[2] <= 1.0, "Random value should be in [0,1] range"
        
        # Test with mathematical function
        v3 = ca.vector(5, ca.mathematical("sin"))
        # Sin values should be accessible
        assert -1.0 <= v3[0] <= 1.0, "Sin value should be in [-1,1] range"
        
        print(" Indexing works with all initialization patterns")
    
    def test_error_messages(self):
        """Test that error messages are descriptive."""
        print("\n=== Testing error messages ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # Test getitem error message
        with pytest.raises(IndexError) as exc_info:
            _ = v[5]
        assert "index out of range" in str(exc_info.value), "getitem error should be descriptive"
        
        # Test setitem error message
        with pytest.raises(IndexError) as exc_info:
            v[5] = 999.0
        assert "index out of range" in str(exc_info.value), "setitem error should be descriptive"
        
        # Test at() error message
        with pytest.raises(IndexError) as exc_info:
            v.at(5)
        assert "index 5 out of range" in str(exc_info.value), "at() error should be descriptive"
        assert "size: 5" in str(exc_info.value), "at() error should show size"
        
        print(" Error messages are descriptive")


if __name__ == "__main__":
    print(" Running indexing operators tests...")
    pytest.main([__file__, "-v"])
