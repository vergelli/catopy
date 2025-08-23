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

    def test_get_devices_returns_list(self):
        """Test that get_devices() returns a list."""
        devices = cato.Devices()
        device_list = devices.get_devices()
        assert isinstance(device_list, list)

    def test_device_list_not_empty(self):
        """Test that at least one CUDA device is detected."""
        devices = cato.Devices()
        device_list = devices.get_devices()
        assert len(device_list) > 0, "No CUDA devices detected"

    def test_device_properties_structure(self):
        """Test that device properties have the expected structure."""
        devices = cato.Devices()
        device_list = devices.get_devices()

        if len(device_list) > 0:
            device = device_list[0]

            assert isinstance(device, dict)

            required_properties = [
                "Name",
                "Compute Capability", 
                "Total Global Memory",
                "Number of SMs",
                "Max Threads Per Block"
            ]

            for prop in required_properties:
                assert prop in device, f"Property '{prop}' not found in device info"

    def test_device_properties_valid_values(self):
        """Test that device properties contain reasonable values."""
        devices = cato.Devices()
        device_list = devices.get_devices()

        if len(device_list) > 0:
            device = device_list[0]

            assert device["Name"].strip() != "", "Device name should not be empty"

            sm_count = int(device["Number of SMs"])
            assert 1 <= sm_count <= 256, f"SM count {sm_count} seems unreasonable"

            memory = int(device["Total Global Memory"])
            assert memory > 0, "Total Global Memory should be positive"

            max_threads = int(device["Max Threads Per Block"])
            assert 32 <= max_threads <= 2048, f"Max threads {max_threads} seems unreasonable"

    def test_get_properties_individual_device(self):
        """Test getting properties for individual devices."""
        devices = cato.Devices()
        device_list = devices.get_devices()

        if len(device_list) > 0:
            device_props = devices.get_properties(0)
            assert isinstance(device_props, dict)
            assert "Name" in device_props

    def test_print_devices_no_exception(self):
        """Test that print_devices() doesn't raise exceptions."""
        devices = cato.Devices()
        try:
            devices.print_devices()
        except Exception as e:
            pytest.fail(f"print_devices() raised an exception: {e}")

    def test_multiple_device_instances(self):
        """Test that multiple Devices instances work correctly."""
        devices1 = cato.Devices()
        devices2 = cato.Devices()

        list1 = devices1.get_devices()
        list2 = devices2.get_devices()
        assert len(list1) == len(list2)

        if len(list1) > 0:
            assert list1[0]["Name"] == list2[0]["Name"]


class TestDeviceEdgeCases:
    """Test class for edge cases and error conditions."""
    
    def test_invalid_device_id(self):
        """Test behavior with invalid device IDs."""
        devices = cato.Devices()

        with pytest.raises(Exception):
            devices.get_properties(999)

    def test_negative_device_id(self):
        """Test behavior with negative device IDs."""
        devices = cato.Devices()

        with pytest.raises(Exception):
            devices.get_properties(-1)


if __name__ == "__main__":
    pytest.main([__file__])
