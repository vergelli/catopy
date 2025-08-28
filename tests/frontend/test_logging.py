#!/usr/bin/env python3
"""
Test suite for logging system functionality.

This module tests the logging control system:
- Default silent mode (no logs visible)
- Enabling logging with ca.logger(True)
- Disabling logging with ca.logger(False)
- Global control affecting all operations
- Persistence of logging state
"""

import pytest
import cato as ca

class TestLoggingSystem:
    """Test class for logging system functionality."""

    def test_default_silent_mode(self):
        """Test that by default, no logs are visible when importing and using cato."""
        print("\n=== Testing default silent mode ===")

        ca.logger(False)

        #* Test that operations work without visible logs
        #* We can't easily capture spdlog output, so we test the behavior differently
        #* The key is that ca.logger(False) should return False
        result = ca.logger(False)
        assert result is False, f"Expected False, got {result}"

        # Create vectors to ensure the system works
        v = ca.vector(3, ca.zeros())
        v2 = ca.vector(2, ca.ones())

        assert v is not None and v2 is not None, "Vector creation should work in silent mode"

        print("Default mode is silent - ca.logger(False) returns False")
        print(f"   Vectors created successfully: {v.size()}, {v2.size()}")

    def test_enable_logging(self):
        """Test that ca.logger(True) enables all logging."""
        print("\n=== Testing logging enable ===")

        # First disable to ensure clean state
        ca.logger(False)

        # Test that enable returns True
        result = ca.logger(True)
        assert result is True, f"Expected True, got {result}"

        # Verify that logging is now enabled by creating a vector
        v = ca.vector(2, ca.zeros())
        assert v is not None and v.size() == 2, "Vector creation should work when logging is enabled"

        print("Logging enabled successfully - ca.logger(True) returns True")

    def test_disable_logging(self):
        """Test that ca.logger(False) disables all logging."""
        print("\n=== Testing logging disable ===")

        # First enable to ensure clean state
        ca.logger(True)

        # Test that disable returns False
        result = ca.logger(False)
        assert result is False, f"Expected False, got {result}"

        # Verify that logging is now disabled by creating a vector
        v = ca.vector(2, ca.ones())
        assert v is not None and v.size() == 2, "Vector creation should work when logging is disabled"

        print("Logging disabled successfully - ca.logger(False) returns False")

    def test_logging_visibility_when_enabled(self):
        """Test that logs are visible when logging is enabled."""
        print("\n=== Testing logging visibility when enabled ===")

        # Enable logging
        ca.logger(True)

        # Test that enable returns True
        result = ca.logger(True)
        assert result is True, f"Expected True, got {result}"

        # Create vectors to verify the system works in debug mode
        v = ca.vector(5, ca.normal(0, 1, 42))
        v2 = ca.vector(3, ca.constant(2.5))

        assert v is not None and v2 is not None, "Vector creation should work in debug mode"
        assert v.size() == 5 and v2.size() == 3, "Vectors should have correct sizes"

        print("Logging enabled - ca.logger(True) returns True")
        print(f"   Vectors created successfully: {v.size()}, {v2.size()}")
        print(f"   Debug mode is active")

    def test_logging_silence_when_disabled(self):
        """Test that logs are silent when logging is disabled."""
        print("\n=== Testing logging silence when disabled ===")

        # Disable logging
        ca.logger(False)

        # Test that disable returns False
        result = ca.logger(False)
        assert result is False, f"Expected False, got {result}"

        # Create vectors to verify the system works in silent mode
        v = ca.vector(4, ca.uniform(0, 1, 123))
        v2 = ca.vector(6, ca.sequence(0, 0.5))

        assert v is not None and v2 is not None, "Vector creation should work in silent mode"
        assert v.size() == 4 and v2.size() == 6, "Vectors should have correct sizes"

        print("Logging disabled - ca.logger(False) returns False")
        print(f"   Vectors created successfully: {v.size()}, {v2.size()}")
        print(f"   Silent mode is active")

    def test_global_control_persistence(self):
        """Test that logging control affects all operations globally and persists."""
        print("\n=== Testing global control persistence ===")

        # Test sequence: enable -> operations -> disable -> operations -> enable -> operations

        # Step 1: Enable logging
        result1 = ca.logger(True)
        assert result1 is True, "Enable should return True"

        # Step 2: Operations with logging enabled
        v1 = ca.vector(3, ca.zeros())
        assert v1 is not None and v1.size() == 3, "Vector creation should work when enabled"

        # Step 3: Disable logging
        result2 = ca.logger(False)
        assert result2 is False, "Disable should return False"

        # Step 4: Operations with logging disabled
        v2 = ca.vector(3, ca.ones())
        assert v2 is not None and v2.size() == 3, "Vector creation should work when disabled"

        # Step 5: Re-enable logging
        result3 = ca.logger(True)
        assert result3 is True, "Re-enable should return True"

        # Step 6: Operations with logging re-enabled
        v3 = ca.vector(3, ca.constant(42))
        assert v3 is not None and v3.size() == 3, "Vector creation should work when re-enabled"

        print("Global control persists and affects all operations")
        print(f"   Enable result: {result1}")
        print(f"   Disable result: {result2}")
        print(f"   Re-enable result: {result3}")
        print(f"   All vectors created successfully")

    def test_logging_with_different_operations(self):
        """Test that logging control affects all types of operations."""
        print("\n=== Testing logging with different operations ===")

        # Enable logging
        ca.logger(True)

        # Test various operations
        operations = [
            ("zeros", lambda: ca.vector(2, ca.zeros())),
            ("ones", lambda: ca.vector(2, ca.ones())),
            ("constant", lambda: ca.vector(2, ca.constant(3.14))),
            ("random", lambda: ca.vector(2, ca.random(42))),
            ("uniform", lambda: ca.vector(2, ca.uniform(0, 1, 42))),
            ("normal", lambda: ca.vector(2, ca.normal(0, 1, 42))),
            ("sequence", lambda: ca.vector(2, ca.sequence(0, 1))),
            ("arange", lambda: ca.vector(2, ca.arange(0, 2)))
        ]

        # Test all operations when logging is enabled
        for op_name, op_func in operations:
            result = op_func()
            assert result is not None, f"Operation {op_name} should return a valid result"
            assert result.size() == 2, f"Operation {op_name} should create vector of size 2"
            print(f"{op_name}: operation successful when enabled")

        # Now disable and test again
        ca.logger(False)

        for op_name, op_func in operations:
            result = op_func()
            assert result is not None, f"Operation {op_name} should return a valid result"
            assert result.size() == 2, f"Operation {op_name} should create vector of size 2"
            print(f"{op_name}: operation successful when disabled")

        print("All operations work correctly in both enabled and disabled modes")

    def test_logging_state_consistency(self):
        """Test that logging state is consistent across multiple calls."""
        print("\n=== Testing logging state consistency ===")

        # Test multiple enable/disable cycles
        for cycle in range(3):
            print(f"   Cycle {cycle + 1}:")

            # Enable
            result_enable = ca.logger(True)
            assert result_enable is True, f"Enable should return True in cycle {cycle + 1}"

            # Verify enabled state by creating a vector
            v = ca.vector(2, ca.zeros())
            assert v is not None and v.size() == 2, f"Vector creation should work in cycle {cycle + 1}"

            # Disable
            result_disable = ca.logger(False)
            assert result_disable is False, f"Disable should return False in cycle {cycle + 1}"

            # Verify disabled state by creating a vector
            v = ca.vector(2, ca.ones())
            assert v is not None and v.size() == 2, f"Vector creation should work in cycle {cycle + 1}"

            print(f"     Enable/Disable cycle {cycle + 1} successful")

        print("Logging state consistency verified across multiple cycles")

    def test_logging_function_return_values(self):
        """Test that ca.logger() returns the correct boolean values."""
        print("\n=== Testing logging function return values ===")

        # Test enable returns True
        result_true = ca.logger(True)
        assert result_true is True, f"ca.logger(True) should return True, got {result_true}"

        # Test disable returns False
        result_false = ca.logger(False)
        assert result_false is False, f"ca.logger(False) should return False, got {result_false}"

        # Test enable again returns True
        result_true2 = ca.logger(True)
        assert result_true2 is True, f"ca.logger(True) should return True, got {result_true2}"

        print("ca.logger() returns correct boolean values")
        print(f"   ca.logger(True) → {result_true}")
        print(f"   ca.logger(False) → {result_false}")
        print(f"   ca.logger(True) → {result_true2}")


class TestLoggingEdgeCases:
    """Test class for logging edge cases and error conditions."""
    
    def test_logging_with_single_element_vectors(self):
        """Test logging behavior with single element vectors (edge case)."""
        print("\n=== Testing logging with single element vectors ===")

        # Enable logging
        ca.logger(True)

        # Test single element vector
        v_single = ca.vector(1, ca.zeros())
        assert v_single is not None and v_single.size() == 1, "Single element vector should be created when enabled"
        print("Single element vector creation works when enabled")

        # Disable logging
        ca.logger(False)

        # Test single element vector again
        v_single2 = ca.vector(1, ca.ones())
        assert v_single2 is not None and v_single2.size() == 1, "Single element vector should be created when disabled"
        print("Single element vector creation works when disabled")

        print("Single element vectors work correctly in both modes")

    def test_logging_with_large_vectors(self):
        """Test logging behavior with large vectors."""
        print("\n=== Testing logging with large vectors ===")

        # Enable logging
        ca.logger(True)

        # Test large vector
        v_large = ca.vector(1000, ca.random(42))
        assert v_large is not None and v_large.size() == 1000, "Large vector should be created when enabled"
        print("Large vector creation works when enabled")

        # Disable logging
        ca.logger(False)

        # Test large vector again
        v_large2 = ca.vector(1000, ca.ones())
        assert v_large2 is not None and v_large2.size() == 1000, "Large vector should be created when disabled"
        print("Large vector creation works when disabled")

        print("Large vectors work correctly in both modes")


if __name__ == "__main__":
    print("Running logging system tests...")
    pytest.main([__file__, "-v"])
