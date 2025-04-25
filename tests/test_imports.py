def test_imports():
    """Test that imports work correctly in different contexts."""
    # Test relative import
    from e3foam.tensors.base import TensorData
    
    # Test that base.py can import utils.py
    tensor_data = TensorData(torch.randn(3, 3))
    projected = tensor_data.project_to_2d()
    
    assert projected is not None, "Projection failed, suggesting import issues"
    
    print("Import test passed!")

if __name__ == "__main__":
    import torch
    test_imports() 