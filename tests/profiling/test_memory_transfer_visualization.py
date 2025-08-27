#!/usr/bin/env python3
"""
Memory Transfer Visualization Test

This test is specifically designed to create clear, visible patterns
in Nsight Systems for analyzing memory transfer operations:
- HOST → GPU transfers (ensure_on_gpu)
- GPU → HOST transfers (ensure_on_host)
- Memory allocation patterns
- Lazy copy behavior
"""

import cato as ca
import time
import gc


def main():
    """Main function for memory transfer visualization."""
    print("Memory Transfer Visualization Test")
    print("This test creates clear patterns for Nsight Systems analysis")
    print("Look for: cudaMemcpy operations, memory allocations, timing patterns")

    # Clear any existing vectors
    gc.collect()

    print("\n=== Phase 1: Vector Creation and Initial GPU Transfer ===")

    # Create vectors of different sizes to see allocation patterns
    sizes = [1000, 10000, 100000]
    vectors = []

    for i, size in enumerate(sizes):
        print(f"   Creating vector {i+1}/3: size {size}")

        # Create vector (stays on HOST initially)
        V = ca.vector(size, ca.constant(i + 1))
        vectors.append(V)

        # Small delay to make operations visible in timeline
        time.sleep(0.1)

        print(f"   Vector {i+1} created, size: {V.size()}")

    print(f"\n   Created {len(vectors)} vectors successfully")

    print("\n=== Phase 2: Sequential GPU Transfers (HOST → GPU) ===")

    # Transfer each vector to GPU sequentially
    for i, V in enumerate(vectors):
        print(f"   Transferring vector {i+1}/3 to GPU...")

        # Force GPU transfer
        V.ensure_on_gpu()

        # Verify transfer
        assert V.is_on_gpu(), f"Vector {i+1} should be on GPU"
        print(f"   Vector {i+1} transferred to GPU")

        # Small delay to make operations visible
        time.sleep(0.2)

    print("   All vectors successfully transferred to GPU")

    print("\n=== Phase 3: Sequential Host Transfers (GPU → HOST) ===")

    # Transfer each vector back to HOST sequentially
    for i, V in enumerate(vectors):
        print(f"   Transferring vector {i+1}/3 back to HOST...")

        # Force HOST transfer
        V.ensure_on_host()

        # Verify transfer
        assert not V.is_on_gpu(), f"Vector {i+1} should be on HOST"
        print(f"   Vector {i+1} transferred to HOST")

        # Small delay to make operations visible
        time.sleep(0.2)

    print("   All vectors successfully transferred to HOST")

    print("\n=== Phase 4: Round-trip Transfers (HOST → GPU → HOST) ===")

    # Test round-trip transfers for each vector
    for i, V in enumerate(vectors):
        print(f"   Round-trip transfer for vector {i+1}/3...")

        # HOST → GPU
        V.ensure_on_gpu()
        assert V.is_on_gpu(), f"Vector {i+1} should be on GPU after round-trip 1"

        # GPU → HOST
        V.ensure_on_host()
        assert not V.is_on_gpu(), f"Vector {i+1} should be on HOST after round-trip 2"

        print(f"   Vector {i+1} round-trip completed")
        time.sleep(0.1)

    print("   All round-trip transfers completed")

    print("\n=== Phase 5: Memory Pressure Test ===")

    # Create many small vectors to see memory allocation patterns
    print("   Creating 20 small vectors for memory pressure...")
    small_vectors = []

    for i in range(20):
        V = ca.vector(1000, ca.constant(i))
        small_vectors.append(V)
        
        if (i + 1) % 5 == 0:
            print(f"   Created {i + 1}/20 small vectors")

    print("   Created 20 small vectors")

    # Transfer all to GPU
    print("   Transferring all small vectors to GPU...")
    for i, V in enumerate(small_vectors):
        V.ensure_on_gpu()
        if (i + 1) % 5 == 0:
            print(f"   Transferred {i + 1}/20 to GPU")

    print("   All small vectors transferred to GPU")

    # Transfer all back to HOST
    print("   Transferring all small vectors back to HOST...")
    for i, V in enumerate(small_vectors):
        V.ensure_on_host()
        if (i + 1) % 5 == 0:
            print(f"   Transferred {i + 1}/20 to HOST")

    print("   All small vectors transferred to HOST")

    print("\n=== Phase 6: Large Vector Stress Test ===")

    # Create one very large vector to see memory allocation
    print("   Creating one very large vector (1M elements)...")
    large_vector = ca.vector(1000000, ca.zeros())
    print("   Large vector created")

    # Transfer to GPU
    print("   Transferring large vector to GPU...")
    large_vector.ensure_on_gpu()
    assert large_vector.is_on_gpu(), "Large vector should be on GPU"
    print("   Large vector transferred to GPU")

    # Transfer back to HOST
    print("   Transferring large vector back to HOST...")
    large_vector.ensure_on_host()
    assert not large_vector.is_on_gpu(), "Large vector should be on HOST"
    print("   Large vector transferred to HOST")

    print("\n=== Phase 7: Final Verification ===")

    # Final verification of all vectors
    all_vectors = vectors + small_vectors + [large_vector]

    print(f"   Total vectors created: {len(all_vectors)}")
    print(f"   Large vectors: {len(vectors)}")
    print(f"   Small vectors: {len(small_vectors)}")
    print(f"   Extra large vector: 1")

    # Clean up
    print("   Cleaning up memory...")
    del all_vectors
    gc.collect()

    print("\nMemory Transfer Visualization Test Completed!")
    print("\nIn Nsight Systems, look for:")
    print("   - cudaMemcpy operations (HOST→GPU and GPU→HOST)")
    print("   - Memory allocation patterns (cudaMalloc)")
    print("   - Timing differences between vector sizes")
    print("   - Sequential vs. parallel transfer patterns")
    print("   - Memory pressure effects")
    print("\nKey timeline markers:")
    print("   - Phase 1: Vector creation (CPU operations)")
    print("   - Phase 2: HOST→GPU transfers (cudaMemcpy)")
    print("   - Phase 3: GPU→HOST transfers (cudaMemcpy)")
    print("   - Phase 4: Round-trip transfers")
    print("   - Phase 5: Memory pressure test")
    print("   - Phase 6: Large vector test")
    print("   - Phase 7: Cleanup")


if __name__ == "__main__":
    main()
