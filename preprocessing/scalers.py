"""
Equivariant scaling operations for tensor data.
"""

from typing import Optional
import torch
from tensors.base import TensorData
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class EquivariantScalerWrapper:
    """
    Wrapper that makes any sklearn-like scaler work with TensorData in an equivariant way.
    Scales tensor fields based on their norms while preserving internal structure.
    
    Example:
        from sklearn.preprocessing import StandardScaler
        scaler = EquivariantScalerWrapper(StandardScaler())
        
        # Fit on training data
        scaled_data = scaler.fit_transform(tensor_data)
        
        # Transform new data
        new_scaled = scaler.transform(new_data)
        
        # Inverse transform
        original_scale = scaler.inverse_transform(scaled_data)
    """
    
    def __init__(self, base_scaler):
        """
        Args:
            base_scaler: Any scaler with fit/transform/inverse_transform interface
        """
        self.base_scaler = base_scaler
        self.tensor_ptr = None  # Store pointer structure for reconstruction
    
    def fit(self, X: TensorData, y: Optional[torch.Tensor] = None):
        """Fit scaler to TensorData norms."""
        self.tensor_ptr = X.ptr  # Remember field structure
        self.base_scaler.fit(X.norms.values)
        return self
        
    def transform(self, X: TensorData) -> TensorData:
        """Scale TensorData by transforming its norms."""
        if not torch.equal(X.ptr.ptr, self.tensor_ptr.ptr):
            raise ValueError("Input tensor structure doesn't match fitted structure")
        scaled_norms = self.base_scaler.transform(X.norms.values)
        return X.apply_norms(torch.from_numpy(scaled_norms).float())
    
    def inverse_transform(self, X: TensorData) -> TensorData:
        """Restore original scale of TensorData."""
        if not torch.equal(X.ptr.ptr, self.tensor_ptr.ptr):
            raise ValueError("Input tensor structure doesn't match fitted structure")
        original_norms = self.base_scaler.inverse_transform(X.norms.values)
        return X.apply_norms(torch.from_numpy(original_norms).float())
    
    def fit_transform(self, X: TensorData, y: Optional[torch.Tensor] = None) -> TensorData:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)


class MeanScaler:
    """
    Simple scaler that divides each field by its mean norm.
    Preserves relative scales between samples while normalizing mean to 1.
    """
    def __init__(self):
        self.coeffs = None
        
    def fit(self, X):
        if isinstance(X, torch.Tensor):
            norms = X
        else:
            norms = torch.from_numpy(X)
        self.coeffs = 1 / torch.mean(norms, dim=0, keepdim=True)
        return self
        
    def transform(self, X):
        if isinstance(X, torch.Tensor):
            norms = X
        else:
            norms = torch.from_numpy(X)
        return (norms * self.coeffs).numpy()
        
    def inverse_transform(self, X):
        if isinstance(X, torch.Tensor):
            norms = X
        else:
            norms = torch.from_numpy(X)
        return (norms / self.coeffs).numpy()

# We'll migrate and improve EquivariantScalerWrap here 

if __name__ == "__main__":
    # Simple test data
    vectors = torch.tensor([
        [2.0, 0.0, 1.0],  # norm ≈ 2.236
        [2.0, 0.0, 0.0],  # norm = 2.0
        [1.0, 1.0, 1.0],  # norm ≈ 1.732
    ])
    
    rank2 = torch.tensor([
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],  # norm = √3
        [[2.0, 0.0, 0.0],
         [0.0, 2.0, 0.0],
         [0.0, 0.0, 2.0]],  # norm = 2√3
        [[0.5, 0.0, 0.0],
         [0.0, 0.5, 0.0],
         [0.0, 0.0, 0.5]]   # norm = √3/2
    ])
    
    # Create and combine fields
    vec_tensor = TensorData(vectors)
    rank2_tensor = TensorData(rank2, symmetry=0)
    combined = TensorData.cat([rank2_tensor, vec_tensor])
    print(combined.tensor.shape)
    
    # Test both scalers
    scalers = {
        "MeanScaler": MeanScaler(),
        "StandardScaler(no center)": StandardScaler(with_mean=False),
    }
    
    for name, base_scaler in scalers.items():
        print(f"\n{name}:")
        scaler = EquivariantScalerWrapper(base_scaler)
        
        # Scale -> Inverse scale
        scaled = scaler.fit_transform(combined)
        recovered = scaler.inverse_transform(scaled)
        
        # Check conservation
        tensor_error = torch.norm(recovered.tensor - combined.tensor)
        print(f"Recovery error: {tensor_error:.8f}")


    '''Conclusions:
    The equivariant scaler must not yield negative values
    
    '''