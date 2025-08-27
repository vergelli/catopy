"""
Pytest configuration for profiling tests
"""

import pytest
import cato

@pytest.fixture(scope="session")
def cato_module():
    """Provide cato module for all profiling tests"""
    return cato

@pytest.fixture
def sample_vector(cato_module):
    """Create a sample vector for testing"""
    return cato_module.vector(100, cato_module.zeros())

@pytest.fixture
def sample_large_vector(cato_module):
    """Create a large sample vector for memory profiling"""
    return cato_module.vector(10000, cato_module.random(42))
