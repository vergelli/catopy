import pytest
import cato as ca

def test_copy_method():
    """Test the copy method functionality"""
    
    # Create original vector
    v = ca.vector(10, ca.constant(5.0))
    print(f"Original vector v: {v}")
    print(f"v[0] = {v[0]}")
    
    # Make a copy using the copy method
    w = v.copy()
    print(f"Copy w: {w}")
    print(f"v[0] = {v[0]}, w[0] = {w[0]}")
    
    # Verify they are equal but not the same object
    print(f"Are v and w equal? {v == w}")
    print(f"Are v and w the same object? {v is w}")
    
    # Modify the copy
    print(f"Modifying w[0] to 999.0...")
    w[0] = 999.0
    
    print(f"After modification:")
    print(f"v: {v}")
    print(f"w: {w}")
    print(f"v[0] = {v[0]}, w[0] = {w[0]}")
    
    # Verify independence
    assert v[0] == 5.0, "Original vector should remain unchanged"
    assert w[0] == 999.0, "Copy should be modified"
    assert v != w, "Vectors should no longer be equal after modification"
    
    print("Copy method test passed!")

def test_copy_vs_assignment():
    """Test copy method vs assignment operator"""
    
    v = ca.vector(5, ca.constant(10.0))
    
    # Assignment (reference)
    w1 = v
    w1[0] = 100.0
    
    # Copy method (independent)
    w2 = v.copy()
    w2[0] = 200.0
    
    print(f"Original v: {v}")
    print(f"Assignment w1: {w1}")
    print(f"Copy w2: {w2}")
    
    # Verify behavior
    assert v[0] == 100.0, "Assignment should modify original"
    assert w1[0] == 100.0, "Assignment reference should be modified"
    assert w2[0] == 200.0, "Copy should be independent"
    
    print("Copy vs assignment test passed!")

if __name__ == "__main__":
    print("=== Testing Copy Method ===")
    test_copy_method()
    
    print("\n=== Testing Copy vs Assignment ===")
    test_copy_vs_assignment()
    
    print("\nAll copy method tests passed!")
