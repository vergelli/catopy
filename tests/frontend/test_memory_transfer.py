"""
Test suite for memory transfer operations in caVectors.

This suite tests:
- ensure_on_gpu() functionality
- ensure_on_host() functionality  
- Dirty flags behavior
- Memory consistency across transfers
- Lazy copy pattern
"""

import pytest
import cato as ca


class TestMemoryTransfer:
    """Test memory transfer operations between HOST and GPU."""
    
    def test_initial_state(self):
        """Test that vectors start on HOST by default."""
        print("\n=== Testing initial state ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # Initially should be on HOST
        assert not v.is_on_gpu(), "Vector should start on HOST"
        assert not v.is_host_dirty(), "HOST should not be dirty initially"
        assert not v.is_gpu_dirty(), "GPU should not be dirty initially"
        
        print(" Initial state: HOST only, no dirty flags")
    
    def test_first_gpu_transfer(self):
        """Test first transfer to GPU."""
        print("\n=== Testing first GPU transfer ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        initial_value = v[0]
        
        # Transfer to GPU
        v.ensure_on_gpu()
        
        # Should now be on GPU
        assert v.is_on_gpu(), "Vector should be on GPU after ensure_on_gpu()"
        assert not v.is_host_dirty(), "HOST should not be dirty after GPU transfer"
        assert not v.is_gpu_dirty(), "GPU should not be dirty after GPU transfer"
        
        # Data should remain consistent
        assert v[0] == initial_value, "Data should remain consistent after GPU transfer"
        
        print(" First GPU transfer successful")
    
    def test_host_modification_marks_gpu_dirty(self):
        """Test that modifying HOST marks GPU as dirty."""
        print("\n=== Testing HOST modification marks GPU dirty ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        v.ensure_on_gpu()  # Transfer to GPU first
        
        # Modify HOST data
        original_value = v[0]
        v[0] = 999.0
        
        # GPU should be marked as dirty
        assert v.is_on_gpu(), "Vector should still be on GPU"
        assert v.is_gpu_dirty(), "GPU should be marked as dirty after HOST modification"
        assert not v.is_host_dirty(), "HOST should not be dirty"
        
        # Data should reflect HOST modification
        assert v[0] == 999.0, "Data should reflect HOST modification"
        
        print(" HOST modification correctly marks GPU as dirty")
    
    def test_gpu_resync_after_host_modification(self):
        """Test that GPU gets resynced after HOST modification."""
        print("\n=== Testing GPU resync after HOST modification ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        v.ensure_on_gpu()  # Transfer to GPU first
        
        # Modify HOST data
        v[0] = 999.0
        v[1] = 888.0
        v[2] = 777.0
        
        # GPU should be dirty
        assert v.is_gpu_dirty(), "GPU should be dirty after multiple HOST modifications"
        
        # Resync GPU
        v.ensure_on_gpu()
        
        # GPU should no longer be dirty
        assert not v.is_gpu_dirty(), "GPU should not be dirty after resync"
        assert not v.is_host_dirty(), "HOST should not be dirty after resync"
        
        # Data should remain consistent
        assert v[0] == 999.0, "Data should remain consistent after GPU resync"
        assert v[1] == 888.0, "Data should remain consistent after GPU resync"
        assert v[2] == 777.0, "Data should remain consistent after GPU resync"
        
        print(" GPU resync successful after HOST modifications")
    
    def test_host_transfer_clears_gpu_flag(self):
        """Test that ensure_on_host() clears GPU flag."""
        print("\n=== Testing ensure_on_host() clears GPU flag ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        v.ensure_on_gpu()  # Transfer to GPU first
        
        # Should be on GPU
        assert v.is_on_gpu(), "Vector should be on GPU"
        
        # Transfer back to HOST
        v.ensure_on_host()
        
        # Should no longer be on GPU (even if GPU buffer exists)
        assert not v.is_on_gpu(), "Vector should not be on GPU after ensure_on_host()"
        
        print(" ensure_on_host() correctly clears GPU flag")
    
    def test_multiple_transfers_efficiency(self):
        """Test that multiple transfers don't cause unnecessary operations."""
        print("\n=== Testing multiple transfers efficiency ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # First transfer to GPU
        v.ensure_on_gpu()
        assert v.is_on_gpu(), "First transfer should work"
        
        # Second transfer to GPU (should be no-op)
        v.ensure_on_gpu()
        assert v.is_on_gpu(), "Second transfer should be no-op"
        
        # Transfer to HOST
        v.ensure_on_host()
        assert not v.is_on_gpu(), "Transfer to HOST should work"
        
        # Transfer to HOST again (should be no-op)
        v.ensure_on_host()
        assert not v.is_on_gpu(), "Second transfer to HOST should be no-op"
        
        print(" Multiple transfers work efficiently")
    
    def test_dirty_flags_consistency(self):
        """Test that dirty flags remain consistent across operations."""
        print("\n=== Testing dirty flags consistency ===")
        
        v = ca.vector(5, ca.constant(3.1415))
        
        # Initial state
        assert not v.is_host_dirty() and not v.is_gpu_dirty(), "Initial state should be clean"
        
        # Transfer to GPU
        v.ensure_on_gpu()
        assert not v.is_host_dirty() and not v.is_gpu_dirty(), "After GPU transfer should be clean"
        
        # Modify HOST
        v[0] = 999.0
        assert not v.is_host_dirty() and v.is_gpu_dirty(), "After HOST modification: HOST clean, GPU dirty"
        
        # Resync GPU
        v.ensure_on_gpu()
        assert not v.is_host_dirty() and not v.is_gpu_dirty(), "After GPU resync should be clean"
        
        print(" Dirty flags remain consistent across operations")
    
    def test_large_vector_transfers(self):
        """Test memory transfers with larger vectors."""
        print("\n=== Testing large vector transfers ===")
        
        # Create larger vector
        v = ca.vector(1000, ca.constant(3.1415))
        
        # Transfer to GPU
        v.ensure_on_gpu()
        assert v.is_on_gpu(), "Large vector should transfer to GPU"
        
        # Modify some elements
        v[0] = 999.0
        v[500] = 888.0
        v[999] = 777.0
        
        # GPU should be dirty
        assert v.is_gpu_dirty(), "GPU should be dirty after modifications"
        
        # Resync GPU
        v.ensure_on_gpu()
        assert not v.is_gpu_dirty(), "GPU should not be dirty after resync"
        
        # Verify data consistency
        assert v[0] == 999.0, "First element should be consistent"
        assert v[500] == 888.0, "Middle element should be consistent"
        assert v[999] == 777.0, "Last element should be consistent"
        
        print(" Large vector transfers work correctly")
    
    def test_transfer_with_different_data_types(self):
        """Test memory transfers with different initialization patterns."""
        print("\n=== Testing transfers with different data types ===")
        
        # Test with zeros
        v1 = ca.vector(10, ca.zeros())
        v1.ensure_on_gpu()
        assert v1.is_on_gpu(), "Zeros vector should transfer to GPU"
        
        # Test with ones
        v2 = ca.vector(10, ca.ones())
        v2.ensure_on_gpu()
        assert v2.is_on_gpu(), "Ones vector should transfer to GPU"
        
        # Test with sequence
        v3 = ca.vector(10, ca.sequence(1, 2))
        v3.ensure_on_gpu()
        assert v3.is_on_gpu(), "Sequence vector should transfer to GPU"
        
        # Test with random
        v4 = ca.vector(10, ca.random(42))
        v4.ensure_on_gpu()
        assert v4.is_on_gpu(), "Random vector should transfer to GPU"
        
        print(" Transfers work with all initialization types")
    
    def test_edge_cases(self):
        """Test edge cases in memory transfers."""
        print("\n=== Testing edge cases ===")
        
        # Single element vector
        v1 = ca.vector(1, ca.constant(42.0))
        v1.ensure_on_gpu()
        assert v1.is_on_gpu(), "Single element should transfer to GPU"
        
        # Two element vector
        v2 = ca.vector(2, ca.constant(3.14))
        v2.ensure_on_gpu()
        assert v2.is_on_gpu(), "Two element should transfer to GPU"
        
        # Modify and transfer back
        v2[0] = 999.0
        v2.ensure_on_host()
        assert not v2.is_on_gpu(), "Should transfer back to HOST"
        
        print(" Edge cases work correctly")


if __name__ == "__main__":
    print(" Running memory transfer tests...")
    pytest.main([__file__, "-v"])
