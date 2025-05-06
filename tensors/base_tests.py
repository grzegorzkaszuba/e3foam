import time
import torch

from e3foam.tensors.base import TensorData
from e3foam.tensors.utils import project_tensor_to_2d


if __name__ == "__main__":
    # Test the TensorData class
    print("Testing TensorData class...")

    # Create a simple tensor field
    tensor = torch.randn(10, 3, 3)
    tensor_data = TensorData(tensor, symmetry=1)
    print(f"Tensor shape: {tensor_data.shape}")
    print(f"Tensor rank: {tensor_data.rank.values}")
    print(f"Tensor symmetry: {tensor_data.symmetry.values}")

    # Test to_irreps and from_irreps
    irrep_tensor, irreps_str, cart_str = tensor_data.to_irreps()
    print(f"Irrep tensor shape: {irrep_tensor.shape}")
    print(f"Irreps string: {irreps_str}")

    reconstructed = TensorData.from_irreps(irrep_tensor, irreps_str, cart_str)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Max difference: {torch.max(torch.abs(tensor_data.tensor - reconstructed.tensor))}")

    # Test with multiple fields
    vector = TensorData(torch.randn(10, 3))
    scalar = TensorData(torch.randn(10, 1))
    combined = TensorData.cat([vector, scalar])
    print(f"Combined shape: {combined.shape}")
    print(f"Combined ptr: {combined.ptr.ptr}")

    # Test the adapters
    print("\n--- Testing Adapters ---")

    # Create a complex tensor field
    batch_size = 100
    complex_tensor = TensorData(torch.randn(batch_size, 3, 3), symmetry=1)
    print(f"Complex tensor shape: {complex_tensor.shape}")

    # Get the adapters
    print("\n--- Creating Complex Tensor Adapters ---")
    complex_irrep_adapter = complex_tensor.get_irrep_adapter()
    complex_cartesian_adapter = complex_tensor.get_cartesian_adapter(cache_rtps=True)

    # Test the standard approach
    print("\n--- Standard TensorData Approach (Complex Tensor) ---")
    start_time = time.time()
    irrep_tensor, irreps_str, cart_str = complex_tensor.to_irreps()
    reconstructed_td = TensorData.from_irreps(irrep_tensor, irreps_str, cart_str)
    standard_time = time.time() - start_time
    print(f"Standard approach time: {standard_time:.6f} seconds")

    # Test the adapter approach
    print("\n--- Adapter Approach (Complex Tensor) ---")
    start_time = time.time()
    irrep_tensor_adapted, irreps_str_adapted = complex_irrep_adapter(complex_tensor.tensor)
    # Pre-compute RTPs by running the adapter once with the sample data
    _ = complex_cartesian_adapter(irrep_tensor_adapted)  # This caches RTPs
    reconstructed_tensor = complex_cartesian_adapter(irrep_tensor_adapted)
    adapter_time = time.time() - start_time
    print(f"Adapter approach time: {adapter_time:.6f} seconds")

    # Compare results
    standard_diff = torch.max(torch.abs(complex_tensor.tensor - reconstructed_td.tensor))
    adapter_diff = torch.max(torch.abs(complex_tensor.tensor - reconstructed_tensor))
    print("\n--- Comparison (Complex Tensor) ---")
    print(f"Standard approach max difference: {standard_diff}")
    print(f"Adapter approach max difference: {adapter_diff}")
    print(f"Results match: {torch.allclose(reconstructed_td.tensor, reconstructed_tensor, atol=1e-5)}")
    print(f"Speed improvement: {standard_time/adapter_time:.2f}x faster")

    # Create a simple vector field
    batch_size = 1000  # Larger batch to better measure performance
    vector_field = TensorData(torch.randn(batch_size, 3))  # Simple 3D vectors
    print(f"Vector field shape: {vector_field.shape}")

    # Get the adapters
    print("\n--- Creating Vector Adapters ---")
    vector_irrep_adapter = vector_field.get_irrep_adapter()
    vector_cartesian_adapter = vector_field.get_cartesian_adapter(cache_rtps=True)

    # Test the standard approach
    print("\n--- Standard TensorData Approach (Vectors) ---")
    start_time = time.time()
    irrep_tensor, irreps_str, cart_str = vector_field.to_irreps()
    reconstructed_td = TensorData.from_irreps(irrep_tensor, irreps_str, cart_str)
    standard_time = time.time() - start_time
    print(f"Standard approach time: {standard_time:.6f} seconds")

    # Test the adapter approach
    print("\n--- Adapter Approach (Vectors) ---")
    start_time = time.time()
    irrep_tensor_adapted, irreps_str_adapted = vector_irrep_adapter(vector_field.tensor)
    # Pre-compute RTPs by running the adapter once
    _ = vector_cartesian_adapter(irrep_tensor_adapted)  # This caches RTPs
    reconstructed_tensor = vector_cartesian_adapter(irrep_tensor_adapted)
    adapter_time = time.time() - start_time
    print(f"Adapter approach time: {adapter_time:.6f} seconds")

    # Compare results
    standard_diff = torch.max(torch.abs(vector_field.tensor - reconstructed_td.tensor))
    adapter_diff = torch.max(torch.abs(vector_field.tensor - reconstructed_tensor))
    print(f"\n--- Comparison (Vectors) ---")
    print(f"Standard approach max difference: {standard_diff}")
    print(f"Adapter approach max difference: {adapter_diff}")
    print(f"Results match: {torch.allclose(reconstructed_td.tensor, reconstructed_tensor, atol=1e-5)}")
    if adapter_time > 0:
        print(f"Speed improvement: {standard_time/adapter_time:.2f}x faster")
    else:
        print(f"Speed improvement: ∞ (adapter time too small to measure)")

    # Test with multiple batches
    print("\n--- Testing with Multiple Vector Batches ---")
    batch_times_standard = []
    batch_times_adapter = []

    for i in range(10):  # More iterations for better measurement
        # Create new random data with same structure
        new_data = torch.randn_like(vector_field.tensor)

        # Standard approach
        start_time = time.time()
        new_td = TensorData(new_data)
        irrep_tensor, irreps_str, cart_str = new_td.to_irreps()
        reconstructed_td = TensorData.from_irreps(irrep_tensor, irreps_str, cart_str)
        batch_times_standard.append(time.time() - start_time)

        # Adapter approach
        start_time = time.time()
        irrep_tensor_adapted = vector_irrep_adapter(new_data)[0]
        reconstructed_tensor = vector_cartesian_adapter(irrep_tensor_adapted)
        batch_times_adapter.append(time.time() - start_time)

        print(f"Batch {i+1}: Results match: {torch.allclose(reconstructed_td.tensor, reconstructed_tensor, atol=1e-5)}, Standard: {batch_times_standard[-1]:.6f}s, Adapter: {batch_times_adapter[-1]:.6f}s")

    # Show average times
    avg_standard = sum(batch_times_standard) / len(batch_times_standard)
    avg_adapter = sum(batch_times_adapter) / len(batch_times_adapter)
    print(f"\nAverage times - Standard: {avg_standard:.6f}s, Adapter: {avg_adapter:.6f}s")
    if avg_adapter > 0:
        print(f"Average speedup: {avg_standard/avg_adapter:.2f}x faster")
    else:
        print(f"Average speedup: ∞ (adapter time too small to measure)")

    # Direct comparison (no TensorData objects at all)
    print("\n--- Direct Function Comparison (No TensorData) ---")

    # Create a direct function that does the same transformation
    def direct_transform(tensor):
        # For vectors, the irrep representation is identical to the Cartesian representation
        return tensor

    # Test with multiple batches
    direct_times = []
    adapter_times = []

    for i in range(10):
        # Create new random data
        new_data = torch.randn_like(vector_field.tensor)

        # Direct approach (just identity function for vectors)
        start_time = time.time()
        result = direct_transform(new_data)
        direct_times.append(time.time() - start_time)

        # Adapter approach (without creating TensorData)
        start_time = time.time()
        irrep_tensor_adapted = vector_irrep_adapter(new_data)[0]
        result_adapter = vector_cartesian_adapter(irrep_tensor_adapted)
        adapter_times.append(time.time() - start_time)

        print(f"Batch {i+1}: Direct: {direct_times[-1]:.6f}s, Adapter: {adapter_times[-1]:.6f}s")

    # Show average times
    avg_direct = sum(direct_times) / len(direct_times)
    avg_adapter = sum(adapter_times) / len(adapter_times)
    print(f"\nAverage times - Direct: {avg_direct:.6f}s, Adapter: {avg_adapter:.6f}s")
    if avg_direct > 0:
        print(f"Adapter overhead: {avg_adapter/avg_direct:.2f}x slower than direct")
    else:
        print(f"Adapter overhead: Cannot calculate (direct time too small to measure)")

    # Test RTP caching explicitly
    print("\n--- Testing RTP Caching ---")

    # Create a new tensor with the same structure
    test_tensor = TensorData(torch.randn(batch_size, 3, 3), symmetry=1)
    test_irrep_adapter = test_tensor.get_irrep_adapter()

    # Create cartesian adapter with caching
    print("Creating adapter with caching...")
    test_cartesian_adapter = test_tensor.get_cartesian_adapter(cache_rtps=True)

        # Convert to irreps
    test_irrep_tensor, _ = test_irrep_adapter(test_tensor.tensor)

    # First run (should compute RTPs)
    print("First run (computing RTPs)...")
    start_time = time.time()
    _ = test_cartesian_adapter(test_irrep_tensor)
    first_time = time.time() - start_time
    print(f"First run time: {first_time:.6f} seconds")

    # Get the cached RTPs
    rtps = test_cartesian_adapter.get_rtps()
    print(f"RTP cache has {len(rtps)} entries")

    # Second run (should use cached RTPs)
    print("Second run (using cached RTPs)...")
    start_time = time.time()
    _ = test_cartesian_adapter(test_irrep_tensor)
    second_time = time.time() - start_time
    print(f"Second run time: {second_time:.6f} seconds")

    # Create a new adapter and set RTPs explicitly
    print("Creating new adapter and setting RTPs explicitly...")
    new_cartesian_adapter = test_tensor.get_cartesian_adapter(cache_rtps=False)
    new_cartesian_adapter.set_rtps(rtps)

    # Run with pre-set RTPs
    print("Run with pre-set RTPs...")
    start_time = time.time()
    _ = new_cartesian_adapter(test_irrep_tensor)
    preset_time = time.time() - start_time
    print(f"Pre-set RTPs run time: {preset_time:.6f} seconds")

    # Test 2D projection performance
    print("\n--- Testing 2D Projection Performance ---")

    # Create test tensors of different types
    batch_size = 1000  # Large batch for better timing

    # Vector field
    vector_field = TensorData(torch.randn(batch_size, 3))
    print(f"Vector field shape: {vector_field.shape}")

    # Symmetric tensor field
    sym_tensor_field = TensorData(torch.randn(batch_size, 3, 3), symmetry=1)
    print(f"Symmetric tensor field shape: {sym_tensor_field.shape}")

    # General tensor field
    gen_tensor_field = TensorData(torch.randn(batch_size, 3, 3))
    print(f"General tensor field shape: {gen_tensor_field.shape}")

    # Test vector projection
    print("\n--- Vector Field Projection ---")

    # Get adapters
    vector_irrep_adapter = vector_field.get_irrep_adapter()
    vector_cartesian_adapter = vector_field.get_cartesian_adapter(cache_rtps=True)
    vector_cartesian_adapter_2d = vector_field.get_cartesian_adapter(
        cache_rtps=True,
        project_2d=True,
        projection_plane='xy'
    )

    # Convert to irreps
    vector_irrep_tensor, _ = vector_irrep_adapter(vector_field.tensor)

    # Time standard approach (without projection)
    print("Standard approach (no projection)...")
    start_time = time.time()
    vector_standard = vector_cartesian_adapter(vector_irrep_tensor)
    standard_time = time.time() - start_time
    print(f"Standard approach time: {standard_time:.6f} seconds")

    # Time with built-in projection
    print("Built-in projection approach...")
    start_time = time.time()
    vector_builtin_proj = vector_cartesian_adapter_2d(vector_irrep_tensor)
    builtin_time = time.time() - start_time
    print(f"Built-in projection time: {builtin_time:.6f} seconds")

    # Time with separate projection
    print("Separate projection approach...")
    start_time = time.time()
    vector_separate_proj = vector_cartesian_adapter.project_to_2d(vector_irrep_tensor)
    separate_time = time.time() - start_time
    print(f"Separate projection time: {separate_time:.6f} seconds")

    # Time direct utility function
    print("Direct utility function approach...")
    start_time = time.time()
    vector_direct_proj = project_tensor_to_2d(vector_standard)
    direct_time = time.time() - start_time
    print(f"Direct utility function time: {direct_time:.6f} seconds")

    # Compare results
    print("\nComparing vector projection results...")
    print(f"Built-in vs Separate: {torch.allclose(vector_builtin_proj, vector_separate_proj)}")
    print(f"Built-in vs Direct: {torch.allclose(vector_builtin_proj, vector_direct_proj)}")

    # Test symmetric tensor projection
    print("\n--- Symmetric Tensor Field Projection ---")

    # Get adapters
    sym_irrep_adapter = sym_tensor_field.get_irrep_adapter()
    sym_cartesian_adapter = sym_tensor_field.get_cartesian_adapter(cache_rtps=True)
    sym_cartesian_adapter_2d = sym_tensor_field.get_cartesian_adapter(
        cache_rtps=True,
        project_2d=True,
        projection_plane='xy',
        preserve_trace=True
    )

    # Convert to irreps
    sym_irrep_tensor, _ = sym_irrep_adapter(sym_tensor_field.tensor)

    # Time standard approach (without projection)
    print("Standard approach (no projection)...")
    start_time = time.time()
    sym_standard = sym_cartesian_adapter(sym_irrep_tensor)
    standard_time = time.time() - start_time
    print(f"Standard approach time: {standard_time:.6f} seconds")

    # Time with built-in projection
    print("Built-in projection approach...")
    start_time = time.time()
    sym_builtin_proj = sym_cartesian_adapter_2d(sym_irrep_tensor)
    builtin_time = time.time() - start_time
    print(f"Built-in projection time: {builtin_time:.6f} seconds")

    # Time with separate projection
    print("Separate projection approach...")
    start_time = time.time()
    sym_separate_proj = sym_cartesian_adapter.project_to_2d(sym_irrep_tensor)
    separate_time = time.time() - start_time
    print(f"Separate projection time: {separate_time:.6f} seconds")

    # Time direct utility function
    print("Direct utility function approach...")
    start_time = time.time()
    sym_direct_proj = project_tensor_to_2d(sym_standard)
    direct_time = time.time() - start_time
    print(f"Direct utility function time: {direct_time:.6f} seconds")

    # Compare results
    print("\nComparing symmetric tensor projection results...")
    print(f"Built-in vs Separate: {torch.allclose(sym_builtin_proj, sym_separate_proj)}")
    print(f"Built-in vs Direct: {torch.allclose(sym_builtin_proj, sym_direct_proj)}")

    # Test trace preservation
    print("\nTesting trace preservation...")

    # For symmetric tensors, we need to handle the flattened format
    # The format is [xx, xy, xz, yy, yz, zz]
    if sym_standard.shape[1] == 6:  # Symmetric tensor in flattened form
        # Calculate trace directly from the flattened format
        standard_trace = sym_standard[:, 0] + sym_standard[:, 3] + sym_standard[:, 5]  # xx + yy + zz
        projected_trace = sym_builtin_proj[:, 0] + sym_builtin_proj[:, 3]  # xx + yy (zz is zeroed)
    else:  # Full tensor format
        # Reshape to 3x3 for easier trace calculation
        sym_standard_3d = sym_standard.view(batch_size, 3, 3)
        sym_builtin_proj_3d = sym_builtin_proj.view(batch_size, 3, 3)

        # Calculate traces
        standard_trace = torch.diagonal(sym_standard_3d, dim1=1, dim2=2).sum(dim=1)
        projected_trace = torch.diagonal(sym_builtin_proj_3d, dim1=1, dim2=2).sum(dim=1)

    # Compare traces
    trace_preserved = torch.allclose(standard_trace, projected_trace)
    print(f"Trace preserved: {trace_preserved}")
    print(f"Max trace difference: {torch.max(torch.abs(standard_trace - projected_trace))}")

    # Test general tensor projection
    print("\n--- General Tensor Field Projection ---")

    # Get adapters
    gen_irrep_adapter = gen_tensor_field.get_irrep_adapter()
    gen_cartesian_adapter = gen_tensor_field.get_cartesian_adapter(cache_rtps=True)
    gen_cartesian_adapter_2d = gen_tensor_field.get_cartesian_adapter(
        cache_rtps=True,
        project_2d=True,
        projection_plane='xy',
        preserve_trace=True
    )

    # Convert to irreps
    gen_irrep_tensor, _ = gen_irrep_adapter(gen_tensor_field.tensor)

    # Time standard approach (without projection)
    print("Standard approach (no projection)...")
    start_time = time.time()
    gen_standard = gen_cartesian_adapter(gen_irrep_tensor)
    standard_time = time.time() - start_time
    print(f"Standard approach time: {standard_time:.6f} seconds")

    # Time with built-in projection
    print("Built-in projection approach...")
    start_time = time.time()
    gen_builtin_proj = gen_cartesian_adapter_2d(gen_irrep_tensor)
    builtin_time = time.time() - start_time
    print(f"Built-in projection time: {builtin_time:.6f} seconds")

    # Time with separate projection
    print("Separate projection approach...")
    start_time = time.time()
    gen_separate_proj = gen_cartesian_adapter.project_to_2d(gen_irrep_tensor)
    separate_time = time.time() - start_time
    print(f"Separate projection time: {separate_time:.6f} seconds")
    
    # Time direct utility function
    print("Direct utility function approach...")
    start_time = time.time()
    gen_direct_proj = project_tensor_to_2d(gen_standard)
    direct_time = time.time() - start_time
    print(f"Direct utility function time: {direct_time:.6f} seconds")
    
    # Compare results
    print("\nComparing general tensor projection results...")
    print(f"Built-in vs Separate: {torch.allclose(gen_builtin_proj, gen_separate_proj)}")
    print(f"Built-in vs Direct: {torch.allclose(gen_builtin_proj, gen_direct_proj)}")
    
    # Test trace preservation for general tensors
    print("\nTesting trace preservation for general tensors...")
    
    # Handle different tensor formats
    if gen_standard.shape[1] == 9:  # Flattened format [xx, xy, xz, yx, yy, yz, zx, zy, zz]
        # Calculate trace directly from the flattened format
        gen_standard_trace = gen_standard[:, 0] + gen_standard[:, 4] + gen_standard[:, 8]  # xx + yy + zz
        gen_projected_trace = gen_builtin_proj[:, 0] + gen_builtin_proj[:, 4]  # xx + yy (zz is zeroed)
    else:  # Full tensor format
        # Reshape to 3x3 for easier trace calculation
        gen_standard_3d = gen_standard.view(batch_size, 3, 3)
        gen_builtin_proj_3d = gen_builtin_proj.view(batch_size, 3, 3)
        
        # Calculate traces
        gen_standard_trace = torch.diagonal(gen_standard_3d, dim1=1, dim2=2).sum(dim=1)
        gen_projected_trace = torch.diagonal(gen_builtin_proj_3d, dim1=1, dim2=2).sum(dim=1)
    
    # Compare traces
    gen_trace_preserved = torch.allclose(gen_standard_trace, gen_projected_trace)
    print(f"Trace preserved: {gen_trace_preserved}")
    print(f"Max trace difference: {torch.max(torch.abs(gen_standard_trace - gen_projected_trace))}")
    
    # Test different projection planes
    print("\n--- Testing Different Projection Planes ---")
    
    # Create a test tensor
    test_tensor = torch.randn(10, 3, 3)
    
    # Project to different planes
    xy_proj = project_tensor_to_2d(test_tensor, plane='xy')
    xz_proj = project_tensor_to_2d(test_tensor, plane='xz')
    yz_proj = project_tensor_to_2d(test_tensor, plane='yz')
    
    # Check that the appropriate components are zero
    print("XY projection - z components zero:", 
          torch.all(xy_proj[:, 2, :] == 0) and torch.all(xy_proj[:, :, 2] == 0))
    print("XZ projection - y components zero:", 
          torch.all(xz_proj[:, 1, :] == 0) and torch.all(xz_proj[:, :, 1] == 0))
    print("YZ projection - x components zero:", 
          torch.all(yz_proj[:, 0, :] == 0) and torch.all(yz_proj[:, :, 0] == 0))
    
    # Test trace preservation across different planes
    print("\nTesting trace preservation across planes...")
    original_trace = torch.diagonal(test_tensor, dim1=1, dim2=2).sum(dim=1)
    xy_trace = torch.diagonal(xy_proj, dim1=1, dim2=2).sum(dim=1)
    xz_trace = torch.diagonal(xz_proj, dim1=1, dim2=2).sum(dim=1)
    yz_trace = torch.diagonal(yz_proj, dim1=1, dim2=2).sum(dim=1)
    
    print(f"Original vs XY: {torch.allclose(original_trace, xy_trace)}")
    print(f"Original vs XZ: {torch.allclose(original_trace, xz_trace)}")
    print(f"Original vs YZ: {torch.allclose(original_trace, yz_trace)}")
    
    # Test without trace preservation
    print("\n--- Testing Without Trace Preservation ---")
    
    # Project without trace preservation
    no_trace_proj = project_tensor_to_2d(test_tensor, plane='xy', preserve_trace=False)
    
    # Check that z components are zero
    print("Z components zero:", 
          torch.all(no_trace_proj[:, 2, :] == 0) and torch.all(no_trace_proj[:, :, 2] == 0))
    
    # Check that trace is not preserved
    no_trace_trace = torch.diagonal(no_trace_proj, dim1=1, dim2=2).sum(dim=1)
    print(f"Trace preserved: {torch.allclose(original_trace, no_trace_trace)}")
    print(f"Max trace difference: {torch.max(torch.abs(original_trace - no_trace_trace))}")
    
    print("\n--- All 2D Projection Tests Completed ---")