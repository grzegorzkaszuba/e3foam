import torch
import pytest
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from e3foam.tensors.base import TensorData
from e3foam.tensors.utils import project_tensor_to_2d

def test_vector_projection():
    """Test 2D projection of vector fields."""
    # Create a batch of random vectors
    batch_size = 10
    vectors = torch.randn(batch_size, 3)
    tensor_data = TensorData(vectors)
    
    # Project to xy plane
    projected = project_tensor_to_2d(vectors, plane='xy')
    
    # Check that z components are zero
    assert torch.all(projected[:, 2] == 0)
    
    # Check that x and y components are unchanged
    assert torch.allclose(projected[:, 0], vectors[:, 0])
    assert torch.allclose(projected[:, 1], vectors[:, 1])
    
    # Test other planes
    projected_xz = project_tensor_to_2d(vectors, plane='xz')
    assert torch.all(projected_xz[:, 1] == 0)  # y should be zero
    
    projected_yz = project_tensor_to_2d(vectors, plane='yz')
    assert torch.all(projected_yz[:, 0] == 0)  # x should be zero

def test_symmetric_tensor_projection():
    """Test 2D projection of symmetric tensor fields."""
    # Create a batch of random symmetric tensors in flattened form [xx, xy, xz, yy, yz, zz]
    batch_size = 10
    tensors = torch.randn(batch_size, 6)
    tensor_data = TensorData(tensors, symmetry=1)
    
    # Project to xy plane with trace preservation
    projected = project_tensor_to_2d(tensors, plane='xy', preserve_trace=True)
    
    # Check that components involving z are zero
    assert torch.all(projected[:, 2] == 0)  # xz
    assert torch.all(projected[:, 4] == 0)  # yz
    assert torch.all(projected[:, 5] == 0)  # zz
    
    # Check trace preservation
    original_trace = tensors[:, 0] + tensors[:, 3] + tensors[:, 5]  # xx + yy + zz
    projected_trace = projected[:, 0] + projected[:, 3]  # xx + yy
    assert torch.allclose(original_trace, projected_trace)
    
    # Test without trace preservation
    projected_no_trace = project_tensor_to_2d(tensors, plane='xy', preserve_trace=False)
    assert torch.all(projected_no_trace[:, 5] == 0)  # zz should be zero
    assert not torch.allclose(original_trace, projected_no_trace[:, 0] + projected_no_trace[:, 3])

def test_general_tensor_projection():
    """Test 2D projection of general tensor fields."""
    # Create a batch of random tensors in matrix form
    batch_size = 10
    tensors = torch.randn(batch_size, 3, 3)
    tensor_data = TensorData(tensors)
    
    # Project to xy plane with trace preservation
    projected = project_tensor_to_2d(tensors, plane='xy', preserve_trace=True)
    
    # Check that z row and column are zero
    assert torch.all(projected[:, 2, :] == 0)
    assert torch.all(projected[:, :, 2] == 0)
    
    # Check trace preservation
    original_trace = torch.diagonal(tensors, dim1=1, dim2=2).sum(dim=1)
    projected_trace = torch.diagonal(projected, dim1=1, dim2=2).sum(dim=1)
    assert torch.allclose(original_trace, projected_trace)

def test_rizzler_projection():
    """Test 2D projection with rizzlers."""
    # Create a batch of random symmetric tensors
    batch_size = 10
    tensors = torch.randn(batch_size, 3, 3)
    # Make them symmetric
    tensors = (tensors + tensors.transpose(1, 2)) / 2
    tensor_data = TensorData(tensors, symmetry=1)
    
    # Get rizzlers
    irrep_rizzler = tensor_data.get_irrep_rizzler()
    
    # Test with projection in constructor
    cartesian_rizzler_2d = tensor_data.get_cartesian_rizzler(
        cache_rtps=True,
        project_2d=True,
        projection_plane='xy',
        preserve_trace=True
    )
    
    # Test with explicit projection method
    cartesian_rizzler = tensor_data.get_cartesian_rizzler(cache_rtps=True)
    
    # Convert to irreps
    irrep_tensor, _ = irrep_rizzler(tensor_data.tensor)
    
    # Get projected results
    projected_constructor = cartesian_rizzler_2d(irrep_tensor)
    projected_method = cartesian_rizzler.project_to_2d(irrep_tensor, plane='xy')
    
    # Check that both approaches give the same result
    assert torch.allclose(projected_constructor, projected_method)
    
    # Check that z components are zero
    assert torch.all(projected_constructor.view(batch_size, 3, 3)[:, 2, :] == 0)
    assert torch.all(projected_constructor.view(batch_size, 3, 3)[:, :, 2] == 0)
    
    # Check trace preservation
    original_trace = torch.diagonal(tensors, dim1=1, dim2=2).sum(dim=1)
    projected_trace = torch.diagonal(projected_constructor.view(batch_size, 3, 3), dim1=1, dim2=2).sum(dim=1)
    assert torch.allclose(original_trace, projected_trace)
    
    # Test different planes
    projected_xz = cartesian_rizzler.project_to_2d(irrep_tensor, plane='xz')
    assert torch.all(projected_xz.view(batch_size, 3, 3)[:, 1, :] == 0)  # y should be zero
    
    projected_yz = cartesian_rizzler.project_to_2d(irrep_tensor, plane='yz')
    assert torch.all(projected_yz.view(batch_size, 3, 3)[:, 0, :] == 0)  # x should be zero

if __name__ == "__main__":
    # Run the tests
    test_vector_projection()
    test_symmetric_tensor_projection()
    test_general_tensor_projection()
    test_rizzler_projection()
    print("All tests passed!") 