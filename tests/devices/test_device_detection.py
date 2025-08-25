"""
Test suite for CUDA device detection functionality.

This module tests the basic device detection capabilities of the cato library,
including device enumeration, property retrieval, and basic validation.
"""

import pytest
import cato


class TestDeviceDetection:
    """Test class for CUDA device detection functionality."""
    
    def test_import_cato(self):
        """Test that cato module can be imported successfully."""
        assert cato is not None

    def test_devices_creation(self):
        """Test that Devices object can be created without errors."""
        devices = cato.Devices()
        assert devices is not None

    def test_devices_count(self):
        """Test that count() returns the number of devices."""
        devices = cato.Devices()
        count = devices.count()
        assert isinstance(count, int)
        assert count > 0, "No CUDA devices detected"

    def test_devices_length(self):
        """Test that len() works on Devices object."""
        devices = cato.Devices()
        assert len(devices) > 0, "No CUDA devices detected"

    def test_get_device_returns_device(self):
        """Test that get_device() returns a Device object."""
        devices = cato.Devices()
        device = devices.get_device(0)
        assert device is not None
        assert hasattr(device, 'get_id')
        assert hasattr(device, 'get_name')

    def test_device_indexing(self):
        """Test that Devices object can be indexed."""
        devices = cato.Devices()
        device = devices[0]
        assert device is not None
        assert hasattr(device, 'get_id')
        assert hasattr(device, 'get_name')

    def test_device_properties_structure(self):
        """Test that device properties have the expected structure."""
        devices = cato.Devices()
        device = devices[0]

        # Test basic device properties
        assert device.get_id() == 0
        assert isinstance(device.get_name(), str)
        assert device.get_name().strip() != "", "Device name should not be empty"

        # Test compute capability
        compute_cap = device.get_compute_capability()
        assert isinstance(compute_cap, str)
        assert "." in compute_cap, "Compute capability should be in format 'major.minor'"

        # Test memory properties
        memory = device.get_total_global_memory()
        assert isinstance(memory, int)
        assert memory > 0, "Total Global Memory should be positive"

        # Test SM count
        sm_count = device.get_multiprocessor_count()
        assert isinstance(sm_count, int)
        assert 1 <= sm_count <= 256, f"SM count {sm_count} seems unreasonable"

        # Test thread properties
        max_threads = device.get_max_threads_per_block()
        assert isinstance(max_threads, int)
        assert 32 <= max_threads <= 2048, f"Max threads {max_threads} seems unreasonable"

    def test_device_properties_valid_values(self):
        """Test that device properties contain reasonable values."""
        devices = cato.Devices()
        device = devices[0]

        # Test device ID
        assert device.get_id() >= 0, "Device ID should be non-negative"

        # Test name
        name = device.get_name()
        assert isinstance(name, str)
        assert len(name) > 0, "Device name should not be empty"
        assert "NVIDIA" in name or "AMD" in name or "Intel" in name, "Device name should contain vendor"

        # Test memory values
        memory = device.get_total_global_memory()
        assert memory > 1024*1024, "Memory should be at least 1MB"
        assert memory < 1024*1024*1024*1024, "Memory should be less than 1TB"

        # Test SM count
        sm_count = device.get_multiprocessor_count()
        assert sm_count > 0, "SM count should be positive"
        assert sm_count <= 256, "SM count should be reasonable"

    def test_device_methods(self):
        """Test that all device methods work correctly."""
        devices = cato.Devices()
        device = devices[0]

        # Test all getter methods
        assert isinstance(device.get_warp_size(), int)
        assert isinstance(device.get_shared_memory_per_block(), int)
        assert isinstance(device.get_registers_per_block(), int)
        assert isinstance(device.get_max_threads_per_multiprocessor(), int)
        assert isinstance(device.get_memory_bus_width(), int)
        assert isinstance(device.get_l2_cache_size(), int)
        assert isinstance(device.get_total_constant_memory(), int)

        # Test tuple methods
        grid_size = device.get_max_grid_size()
        assert isinstance(grid_size, tuple)
        assert len(grid_size) == 3
        assert all(isinstance(x, int) for x in grid_size)

        threads_dim = device.get_max_threads_dim()
        assert isinstance(threads_dim, tuple)
        assert len(threads_dim) == 3
        assert all(isinstance(x, int) for x in threads_dim)

    def test_device_show_method(self):
        """Test that show() method works without exceptions."""
        devices = cato.Devices()
        device = devices[0]
        
        try:
            device.show()
        except Exception as e:
            pytest.fail(f"device.show() raised an exception: {e}")

    def test_devices_show_method(self):
        """Test that Devices.show() method works without exceptions."""
        devices = cato.Devices()
        
        try:
            devices.show()
        except Exception as e:
            pytest.fail(f"devices.show() raised an exception: {e}")

    def test_device_get_properties(self):
        """Test that get_properties() returns a dictionary."""
        devices = cato.Devices()
        device = devices[0]
        
        props = device.get_properties()
        assert isinstance(props, dict)
        assert len(props) > 0
        
        # Check that all required properties are present
        required_properties = [
            "Device ID", "Name", "Compute Capability", "Total Global Memory",
            "Shared Memory Per Block", "Registers Per Block", "Warp Size",
            "Max Threads Per Block", "Max Threads Per MultiProcessor",
            "Number of SMs", "Memory Bus Width (bits)", "L2 Cache Size",
            "Max Grid Size", "Max Threads Dim", "Total Constant Memory"
        ]
        
        for prop in required_properties:
            assert prop in props, f"Property '{prop}' not found in device properties"

    def test_multiple_device_instances(self):
        """Test that multiple Devices instances work correctly."""
        devices1 = cato.Devices()
        devices2 = cato.Devices()

        assert len(devices1) == len(devices2)

        if len(devices1) > 0:
            device1 = devices1[0]
            device2 = devices2[0]
            assert device1.get_name() == device2.get_name()
            assert device1.get_id() == device2.get_id()

    def test_device_string_representation(self):
        """Test that Device objects have proper string representation."""
        devices = cato.Devices()
        device = devices[0]
        
        # Test __str__
        device_str = str(device)
        assert isinstance(device_str, str)
        assert "Device(id=" in device_str
        assert device.get_name() in device_str

    def test_devices_string_representation(self):
        """Test that Devices objects have proper string representation."""
        devices = cato.Devices()

        # Test __str__
        devices_str = str(devices)
        assert isinstance(devices_str, str)
        assert "Devices(count=" in devices_str
        assert str(len(devices)) in devices_str



class TestDeviceEdgeCases:
    """Test class for edge cases and error conditions."""
    
    def test_invalid_device_id(self):
        """Test behavior with invalid device IDs."""
        devices = cato.Devices()

        with pytest.raises(Exception):
            devices.get_device(999)

    def test_negative_device_id(self):
        """Test behavior with negative device IDs."""
        devices = cato.Devices()

        with pytest.raises(Exception):
            devices.get_device(-1)

    def test_index_out_of_range(self):
        """Test behavior with index out of range."""
        devices = cato.Devices()
        device_count = len(devices)

        with pytest.raises(Exception):
            _ = devices[device_count]  # Should fail

    def test_empty_devices_list(self):
        """Test behavior when no devices are available."""
        # This test might not be applicable if CUDA is required
        # but we can test the error handling
        pass


if __name__ == "__main__":
    pytest.main([__file__])
