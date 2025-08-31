import pytest
import cato as ca

class TestSlicing:
    """Test suite for caVector slicing operations"""
    
    def setup_method(self):
        """Setup test vector before each test"""
        self.vec = ca.vector(20, ca.constant(5))
        # Set some specific values for testing
        for i in range(5):
            self.vec[i] = i  # [0, 1, 2, 3, 4, 5, 5, 5, ...]

    def test_basic_slicing(self):
        """Test basic slicing operations"""
        # Basic slice
        result = self.vec[0:5]
        assert result.size() == 5
        assert result[0] == 0
        assert result[4] == 4

        # Slice with step
        result = self.vec[0:10:2]
        assert result.size() == 5
        assert result[0] == 0
        assert result[1] == 2
        assert result[2] == 4
        assert result[3] == 5
        assert result[4] == 5
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Empty slice
        result = self.vec[5:5]
        assert result.size() == 0

        # Single element slice
        result = self.vec[5:6]
        assert result.size() == 1
        assert result[0] == 5

        # Slice beyond bounds
        result = self.vec[15:25]
        assert result.size() == 5
        assert result[0] == 5

        # Slice from beyond bounds
        result = self.vec[25:30]
        assert result.size() == 0

    def test_negative_step(self):
        """Test reverse slicing with negative step"""
        # Reverse slice
        result = self.vec[10:0:-1]
        assert result.size() == 10
        assert result[0] == 5  # First element of result
        assert result[9] == 1  # Last element of result (index 1 from original)

        # Reverse slice with step
        result = self.vec[15:5:-2]
        assert result.size() == 5
        assert result[0] == 5
        assert result[1] == 5
        assert result[2] == 5
        assert result[3] == 5
        assert result[4] == 5

    def test_default_parameters(self):
        """Test slicing with default start/stop/step"""
        # Default start (0)
        result = self.vec[:5]
        assert result.size() == 5
        assert result[0] == 0

        # Default stop (size)
        result = self.vec[15:]
        assert result.size() == 5
        assert result[0] == 5

        # Default step (1)
        result = self.vec[0:5:]
        assert result.size() == 5
        assert result[0] == 0
        assert result[4] == 4

    def test_independence(self):
        """Test that sliced vectors are independent"""
        original = self.vec[0:5]
        modified = self.vec[0:5]

        # Modify the slice
        modified[0] = 999

        # Original should remain unchanged
        assert original[0] == 0
        assert modified[0] == 999

        # Original vector should remain unchanged
        assert self.vec[0] == 0

    def test_large_step(self):
        """Test slicing with large step values"""
        # Large step
        result = self.vec[0:20:5]
        assert result.size() == 4
        assert result[0] == 0
        assert result[1] == 5
        assert result[2] == 5
        assert result[3] == 5

        # Step larger than slice
        result = self.vec[0:5:10]
        assert result.size() == 1
        assert result[0] == 0

    def test_zero_step_error(self):
        """Test that zero step raises error"""
        with pytest.raises(Exception):  # Should raise std::invalid_argument
            self.vec[0:10:0]

    def test_memory_consistency(self):
        """Test memory consistency after slicing"""
        # Ensure original vector is still accessible
        assert self.vec.size() == 20
        assert self.vec[0] == 0
        assert self.vec[19] == 5

        # Multiple slices should work
        slice1 = self.vec[0:5]
        slice2 = self.vec[5:10]
        slice3 = self.vec[10:15]

        assert slice1.size() == 5
        assert slice2.size() == 5
        assert slice3.size() == 5

        # All slices should be independent
        slice1[0] = 100
        slice2[0] = 200
        slice3[0] = 300

        assert self.vec[0] == 0  # Original unchanged
        assert self.vec[5] == 5  # Original unchanged
        assert self.vec[10] == 5  # Original unchanged

    def test_full_reverse_slice(self):
        """Test the special case v[::-1] - reverse entire vector"""
        # Test with our special case detection
        result = self.vec[::-1]
        # Vector has 20 elements: [0,1,2,3,4,5,5,5,...,5] -> reversed
        expected = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0]
        assert result.size() == 20
        # Check first few and last few elements to avoid long list comparison
        assert result[0] == 5   # Last element of original
        assert result[15] == 4  # Element at index 4 of original
        assert result[16] == 3  # Element at index 3 of original
        assert result[17] == 2  # Element at index 2 of original
        assert result[18] == 1  # Element at index 1 of original
        assert result[19] == 0  # First element of original
        
        # Verify it's a deep copy (independent)
        result[0] = 999
        assert self.vec[19] == 5  # Original unchanged
        assert result[0] == 999  # Copy modified

    def test_edge_case_slicing(self):
        """Test edge cases that could break slicing"""
        # Empty slice from middle
        result = self.vec[5:5]
        assert result.size() == 0
        
        # Single element slice
        result = self.vec[3:4]
        assert result.size() == 1
        assert result[0] == 3
        
        # Slice with step larger than vector
        result = self.vec[0::15]
        assert result.size() == 2
        assert result[0] == 0
        assert result[1] == 5   # 0 + 15 = 15 (within bounds, but value is 5 from constant)
        
        # Negative step with small range
        result = self.vec[2:0:-1]
        assert result.size() == 2
        assert result[0] == 2
        assert result[1] == 1

    def test_boundary_conditions(self):
        """Test boundary conditions that could cause issues"""
        # Slice at exact boundaries
        result = self.vec[0:20]
        assert result.size() == 20
        
        # Slice with negative indices
        result = self.vec[-5:-1]
        assert result.size() == 4
        assert result[0] == 5  # index 15
        assert result[3] == 5  # index 18
        
        # Slice with mixed positive/negative
        result = self.vec[0:-1]
        assert result.size() == 19
        assert result[0] == 0
        assert result[18] == 5  # index 18

if __name__ == "__main__":
    # Run tests manually
    test = TestSlicing()
    test.setup_method()

    print("Running slicing tests...")

    test.test_basic_slicing()
    print("Basic slicing passed")

    test.test_edge_cases()
    print("Edge cases passed")

    test.test_negative_step()
    print("Negative step passed")

    test.test_default_parameters()
    print("Default parameters passed")

    test.test_independence()
    print("Independence test passed")

    test.test_large_step()
    print("Large step passed")

    test.test_memory_consistency()
    print("Memory consistency passed")

    test.test_full_reverse_slice()
    print("Full reverse slice passed")

    test.test_edge_case_slicing()
    print("Edge case slicing passed")

    test.test_boundary_conditions()
    print("Boundary conditions passed")

    print("\nAll slicing tests passed!")
