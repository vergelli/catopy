"""
Pytest configuration file for catopy tests.

This file contains shared fixtures and configuration for all tests.
"""

import pytest
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def cuda_available():
    """
    Fixture to check if CUDA is available on the system.

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        import cato
        devices = cato.Devices()
        device_list = devices.get_devices()
        return len(device_list) > 0
    except Exception:
        return False


@pytest.fixture(scope="session") 
def sample_device():
    """
    Fixture that provides a sample CUDA device for testing.

    Returns:
        dict: Device properties of the first available CUDA device
    """
    try:
        import cato
        devices = cato.Devices()
        device_list = devices.get_devices()
        if len(device_list) > 0:
            return device_list[0]
        else:
            pytest.skip("No CUDA devices available for testing")
    except Exception as e:
        pytest.skip(f"Could not initialize CUDA devices: {e}")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA hardware"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests that require CUDA."""
    for item in items:
        if "cuda" in item.nodeid.lower() or "device" in item.nodeid.lower():
            item.add_marker(pytest.mark.cuda)
