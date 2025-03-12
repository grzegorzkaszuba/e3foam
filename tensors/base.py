"""
Core tensor operations and data structures for handling physical fields with equivariance properties.

This module provides a framework for handling tensors of different ranks while preserving their
physical properties and symmetries. It supports both single-field tensors (like a velocity field)
and multi-field tensors (like combined velocity and stress fields).

Key Concepts:
    1. Physical Fields:
       - Each tensor represents one or more physical fields
       - Fields maintain their physical properties (rank, symmetry, parity)
       - Fields can be combined while preserving their individual properties

    2. Tensor Properties:
       - Rank: The order of the tensor (0: scalar, 1: vector, 2: rank-2 tensor)
       - Symmetry: Symmetry properties of rank-2 tensors (0: none, 1: symmetric, -1: skew-symmetric)
       - Parity: Transformation behavior under reflection (-1: odd, 1: even)

    3. Field Pointers:
       - Track boundaries between different physical fields
       - Enable operations on individual fields within multi-field tensors
       - Preserve physical meaning during tensor operations

Example Usage:
    # Create a single velocity field (rank-1 tensor)
    velocity = TensorData(velocity_tensor)  # shape: [batch_size, 3]
    
    # Create a stress tensor (rank-2, symmetric)
    stress = TensorData(stress_tensor, symmetry=1)  # shape: [batch_size, 6]
    
    # Combine fields while preserving their properties
    combined = TensorData.cat([velocity, stress])  # shape: [batch_size, 9]
    # combined.ptr.ptr shows [0, 3, 9] - separating velocity and stress components
"""

from dataclasses import dataclass
import copy
import torch
from typing import List
from e3nn.io import CartesianTensor
from e3nn import o3  # Import here to avoid circular imports


def safe_divide(numerator, denominator, safe_value=0):
    """Safe division handling zeros in denominator."""
    # Extract values if TensorIndex objects are passed
    if isinstance(numerator, TensorIndex):
        numerator = numerator.values
    if isinstance(denominator, TensorIndex):
        denominator = denominator.values
        
    safe_denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    result = numerator / safe_denominator
    return torch.where(denominator == 0, torch.full_like(result, safe_value), result)


class TensorData:
    """
    A class representing tensor data with equivariance properties.
    
    This class handles both single-field and multi-field tensors, maintaining their physical
    properties and relationships. It provides functionality for:
    1. Automatic rank and symmetry inference for single fields
    2. Explicit property handling for multi-field tensors
    3. Component-wise operations preserving tensor properties
    4. Field concatenation and stacking operations
    
    Key Properties:
        tensor: The underlying tensor data, always in 2D form [batch_size, features]
        ptr: TensorIndex tracking field boundaries in the feature dimension
        rank: TensorIndex storing rank values for each field
        symmetry: TensorIndex storing symmetry values for each field
        parity: TensorIndex storing parity values for each field
    
    Field Properties (tracked by TensorIndex):
        Rank:
            0: Scalar field (1 component)
            1: Vector field (3 components)
            2: Rank-2 tensor field (3x3 components, may be reduced by symmetry)
        
        Symmetry (for rank-2 tensors):
            0: No symmetry (9 components)
            1: Symmetric (6 unique components)
            -1: Skew-symmetric (3 unique components)
        
        Parity:
            1: Even (unchanged under reflection)
            -1: Odd (sign changes under reflection)
    
    Example:
        # Single vector field
        vector = TensorData(torch.randn(100, 3))  # 100 samples, 3 components
        print(vector.rank.values)  # tensor([1])
        print(vector.ptr.ptr)      # tensor([0, 3])
        
        # Multi-field tensor (vector + symmetric tensor)
        v1 = TensorData(torch.randn(100, 3))           # vector
        t1 = TensorData(torch.randn(100, 3, 3), symmetry=1)  # symmetric tensor
        combined = TensorData.cat([v1, t1])
        print(combined.ptr.ptr)  # tensor([0, 3, 9]) - separating fields
    """
    
    def __init__(self, tensor: torch.Tensor, parity: int = -1, rank: int = None, symmetry: int = 0, 
                 is_multi_field: bool = False, is_flattened: bool = False):
        """
        Initialize TensorData with a tensor and optional properties.
        
        Args:
            tensor: Input tensor
            parity: Tensor parity (-1 for odd, +1 for even)
            rank: Tensor rank
            symmetry: Tensor symmetry (0: none, 1: symmetric, -1: skew-symmetric)
            is_multi_field: If True, skips rank inference
            is_flattened: If True, assumes symmetric/skew tensors are already flattened
        """
        # Convert tensor to float32 if needed
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.float64:
                self.tensor = tensor.to(torch.float32)
            else:
                self.tensor = tensor
        else:
            self.tensor = torch.tensor(tensor, dtype=torch.float32)

        if is_multi_field:
            # For multi-field tensors, just store the tensor and wait for explicit property setting
            self.tensor = self.tensor.reshape(self.tensor.shape[0], -1)
            return

        # Rest of the initialization for single-field tensors
        inferred_rank = self._infer_rank() if rank is None else rank
        
        # Initialize structural indices with correct size based on rank and symmetry
        if inferred_rank == 2:
            n_components = {
                0: 9,  # No symmetry
                1: 6,  # Symmetric
                -1: 3  # Skew-symmetric
            }[symmetry]
        else:
            n_components = 1 if inferred_rank == 0 else 3
        
        self.ptr = TensorIndex.empty(n_components)
        
        # Create rank and symmetry indices
        self.rank = TensorIndex(
            values=torch.tensor([inferred_rank], dtype=torch.long),
            ptr=torch.tensor([0, n_components], dtype=torch.long),
            dtype=torch.long
        )
        
        self.symmetry = TensorIndex(
            values=torch.tensor([symmetry], dtype=torch.long),
            ptr=torch.tensor([0, n_components], dtype=torch.long),
            dtype=torch.long
        )
        
        # Set parity index (either from input or inferred from rank)
        if parity not in [-1, 1]:
            parity = 1 if inferred_rank % 2 == 0 else -1
        self.parity = TensorIndex(
            values=torch.tensor([parity], dtype=torch.long),
            ptr=torch.tensor([0, self.tensor.shape[-1]], dtype=torch.long),
            dtype=torch.long
        )
        
        # Skip flattening if already in correct format
        if not is_flattened:
            self.flatten_to_unique()

    def _infer_rank(self) -> int:
        """Infer tensor rank from shape."""
        shape = self.tensor.shape
        values_per_field = 1
        for d in shape[1:]:
            values_per_field = values_per_field * d
            
        if values_per_field == 1:
            return 0
        elif values_per_field == 3:
            return 1
        elif values_per_field == 9:
            return 2
        else:
            raise ValueError(f"Cannot infer rank from shape {shape}")

    def flatten_to_unique(self):
        """Convert tensor to 2D with only unique components."""
        if self.rank.values[0] == 2:
            batch_size = self.tensor.shape[0]
            symmetry = self.symmetry.values[0].item()
            
            if symmetry == 0:  # No symmetry - just reshape
                self.tensor = self.tensor.reshape(batch_size, 9)
            elif symmetry == 1:  # Symmetric - optimize storage
                # Take upper triangular part [xx, xy, xz, yy, yz, zz]
                unique = torch.zeros((batch_size, 6), dtype=self.tensor.dtype)
                unique[:, 0] = self.tensor[:, 0, 0]  # xx
                unique[:, 1] = self.tensor[:, 0, 1]  # xy
                unique[:, 2] = self.tensor[:, 0, 2]  # xz
                unique[:, 3] = self.tensor[:, 1, 1]  # yy
                unique[:, 4] = self.tensor[:, 1, 2]  # yz
                unique[:, 5] = self.tensor[:, 2, 2]  # zz
                self.tensor = unique
            elif symmetry == -1:  # Skew-symmetric - optimize storage
                # Take upper triangular part [xy, xz, yz]
                unique = torch.zeros((batch_size, 3), dtype=self.tensor.dtype)
                unique[:, 0] = self.tensor[:, 0, 1]  # xy
                unique[:, 1] = self.tensor[:, 0, 2]  # xz
                unique[:, 2] = self.tensor[:, 1, 2]  # yz
                self.tensor = unique
        elif self.rank.values[0] == 0:
            # Scalar - already in correct form (batch_size, 1)
            self.tensor = self.tensor.reshape(-1, 1)
        elif self.rank.values[0] == 1:
            # Vector - all components are unique (batch_size, 3)
            self.tensor = self.tensor.reshape(-1, 3)
        else:
            raise ValueError(f"Unknown rank: {self.rank.values[0]}")

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def num_fields(self) -> int:
        """Number of physical fields in the tensor."""
        return len(self.ptr.ptr) - 1

    @property
    def norms(self):
        """Compute the norms of each field in the tensor."""
        values = []
        # Calculate norm for each field separately using ptr boundaries
        for i in range(self.num_fields):
            start, end = self.ptr.ptr[i], self.ptr.ptr[i+1]
            field_norm = torch.norm(self.tensor[:, start:end], dim=1, keepdim=True)
            values.append(field_norm)
        
        # Combine norms from all fields
        values = torch.cat(values, dim=1)
        
        # Create pointer structure for norms (one norm per field)
        norm_ptr = torch.arange(len(self.ptr.ptr), dtype=torch.long)
        
        return TensorIndex(
            values=values,
            ptr=norm_ptr,
            dtype=values.dtype
        )

    def apply_norms(self, norms, inplace=False):
        """Apply new norms to the tensor while preserving its direction."""
        if not inplace:
            return copy.deepcopy(self).apply_norms(norms, inplace=True)
        
        current_norms = self.norms.values  # Extract values here
        if isinstance(norms, TensorIndex):
            norms = norms.values  # Extract values if TensorIndex is passed
        
        scale_factors = safe_divide(norms, current_norms)
        num_repeats = self.ptr.ptr[1:]-self.ptr.ptr[:-1] # this returns 1 dimensional tensor: 6, 3, 3
        repeated_scale_factors = torch.repeat_interleave(scale_factors, num_repeats, dim=1)
        new_tensor = repeated_scale_factors * self.tensor  # Simplified reshape
        self.tensor = new_tensor
        return self

    @classmethod
    def cat(cls, tensors: List['TensorData']) -> 'TensorData':
        """
        Concatenate TensorData objects along feature dimension.
        Maintains batch size while concatenating features and updating pointers.
        
        Args:
            tensors: List of TensorData objects with same batch size
        """
        # Verify batch sizes match
        batch_size = tensors[0].tensor.shape[0]
        assert all(t.tensor.shape[0] == batch_size for t in tensors), "All tensors must have same batch size"
        
        # Concatenate tensors
        cat_tensor = torch.cat([t.tensor for t in tensors], dim=1)
        
        # Create new pointers based on cumulative feature counts
        ptr = [0]
        for t in tensors:
            ptr.append(ptr[-1] + t.tensor.shape[1])
        ptr = torch.tensor(ptr, dtype=torch.long)
        
        # Combine ranks, symmetries, and parities
        ranks = torch.cat([t.rank.values for t in tensors])
        symmetries = torch.cat([t.symmetry.values for t in tensors])
        parities = torch.cat([t.parity.values for t in tensors])
        
        # Create new TensorData with combined properties
        result = cls(cat_tensor, is_multi_field=True)
        result.ptr = TensorIndex(torch.tensor([]), ptr, dtype=torch.long)
        result.rank = TensorIndex(ranks, ptr, dtype=torch.long)
        result.symmetry = TensorIndex(symmetries, ptr, dtype=torch.long)
        result.parity = TensorIndex(parities, ptr, dtype=torch.long)
        
        return result

    def append(self, other: 'TensorData') -> 'TensorData':
        """
        Append another TensorData along batch dimension (increasing number of examples).
        Requires exact matching of tensor structure and properties.
        
        Args:
            other: TensorData object with identical field structure
            
        Returns:
            New TensorData with combined examples
            
        Raises:
            ValueError: If tensor structures don't match exactly
        """
        # Validate structural properties match exactly
        if not torch.equal(self.ptr.ptr, other.ptr.ptr):
            raise ValueError("Field pointers must match exactly")
        
        if not torch.equal(self.rank.values, other.rank.values):
            raise ValueError("Rank values must match exactly")
        
        if not torch.equal(self.symmetry.values, other.symmetry.values):
            raise ValueError("Symmetry values must match exactly")
        
        if not torch.equal(self.parity.values, other.parity.values):
            raise ValueError("Parity values must match exactly")
        
        # If all validations pass, combine the tensors
        combined_tensor = torch.cat([self.tensor, other.tensor], dim=0)
        
        # Create new TensorData with same properties
        result = TensorData(combined_tensor, is_multi_field=True)
        result.ptr = self.ptr
        result.rank = self.rank
        result.symmetry = self.symmetry
        result.parity = self.parity
        
        return result

    def get_field(self, field_idx: int) -> 'TensorData':
        """
        Extract a single physical field from a multi-field tensor.
        
        Args:
            field_idx: Index of the field to extract (0-based)
            
        Returns:
            TensorData containing only the requested field
            
        Raises:
            IndexError: If field_idx is out of range
        """
        if field_idx >= len(self.ptr.ptr) - 1:
            raise IndexError(f"Field index {field_idx} out of range. Tensor has {len(self.ptr.ptr) - 1} fields")
        
        # Get field boundaries
        start, end = self.ptr.ptr[field_idx], self.ptr.ptr[field_idx + 1]
        
        # Extract field data
        field_tensor = self.tensor[:, start:end]
        
        # Create new TensorData with single field properties
        result = TensorData(field_tensor, is_multi_field=True)
        result.ptr = TensorIndex.empty(end - start)
        result.rank = TensorIndex(
            values=self.rank.values[field_idx:field_idx+1],
            ptr=torch.tensor([0, end - start]),
            dtype=torch.long
        )
        result.symmetry = TensorIndex(
            values=self.symmetry.values[field_idx:field_idx+1],
            ptr=torch.tensor([0, end - start]),
            dtype=torch.long
        )
        result.parity = TensorIndex(
            values=self.parity.values[field_idx:field_idx+1],
            ptr=torch.tensor([0, end - start]),
            dtype=torch.long
        )
        
        return result

    def to_irreps(self):
        """
        Convert TensorData to e3nn representation.
        Returns:
            tuple: (transformed_tensor, irreps_string)
            - transformed_tensor: Tensor in irrep basis
            - irreps_string: e3nn irreps description
        """
        irrep_tensors = []
        irrep_strs = []
        
        for i in range(self.num_fields):
            field = self.get_field(i)
            rank = field.rank.values[0].item()
            symmetry = field.symmetry.values[0].item()
            
            if rank == 0:  # Scalar
                irrep_tensors.append(field.tensor)
                irrep_strs.append("1x0e")
                
            elif rank == 1:  # Vector
                irrep_tensors.append(field.tensor)
                # TODO: Consider mechanism to handle '1e' vectors based on domain knowledge
                irrep_strs.append("1x1o")
                
            elif rank == 2:  # Rank-2 tensor
                # Create full tensor with same device/dtype as input
                batch_size = field.tensor.shape[0]
                full_tensor = torch.zeros(
                    (batch_size, 3, 3), 
                    device=field.tensor.device, 
                    dtype=field.tensor.dtype
                )
                
                if symmetry == 0:  # No symmetry
                    full_tensor = field.tensor.reshape(batch_size, 3, 3)
                    ct = CartesianTensor('ij')
                    irrep_tensors.append(ct.from_cartesian(full_tensor))
                    irrep_strs.append(str(o3.Irreps(ct)))
                    
                elif symmetry == 1:  # Symmetric
                    # [xx, xy, xz, yy, yz, zz] -> 3x3
                    full_tensor[:, 0, 0] = field.tensor[:, 0]  # xx
                    full_tensor[:, 0, 1] = field.tensor[:, 1]  # xy
                    full_tensor[:, 1, 0] = field.tensor[:, 1]  # xy
                    full_tensor[:, 0, 2] = field.tensor[:, 2]  # xz
                    full_tensor[:, 2, 0] = field.tensor[:, 2]  # xz
                    full_tensor[:, 1, 1] = field.tensor[:, 3]  # yy
                    full_tensor[:, 1, 2] = field.tensor[:, 4]  # yz
                    full_tensor[:, 2, 1] = field.tensor[:, 4]  # yz
                    full_tensor[:, 2, 2] = field.tensor[:, 5]  # zz
                    
                    sct = CartesianTensor('ij=ji')
                    irrep_tensors.append(sct.from_cartesian(full_tensor))
                    irrep_strs.append(str(o3.Irreps(sct)))
                    
                elif symmetry == -1:  # Skew-symmetric
                    # [xy, xz, yz] -> 3x3
                    full_tensor[:, 0, 1] = field.tensor[:, 0]   # xy
                    full_tensor[:, 1, 0] = -field.tensor[:, 0]  # -xy
                    full_tensor[:, 0, 2] = field.tensor[:, 1]   # xz
                    full_tensor[:, 2, 0] = -field.tensor[:, 1]  # -xz
                    full_tensor[:, 1, 2] = field.tensor[:, 2]   # yz
                    full_tensor[:, 2, 1] = -field.tensor[:, 2]  # -yz
                    
                    act = CartesianTensor('ij=-ji')
                    irrep_tensors.append(act.from_cartesian(full_tensor))
                    irrep_strs.append(str(o3.Irreps(act)))
        
        # Combine tensors and strings
        combined_tensor = torch.cat(irrep_tensors, dim=-1)
        # Create a single Irreps object from all strings and convert back to string
        irreps_str = str(o3.Irreps(" + ".join(irrep_strs)))
        
        return combined_tensor, irreps_str

@dataclass
class TensorIndex:
    """
    Maps tensor properties (like norms) to their components using pointer-style indexing.
    Similar to torch_geometric's batch.ptr, indicates where different tensor components begin and end.
    
    Example:
        For a batch of vectors (rank 1 tensors), each with norm N:
        values = [N1, N2, N3]  # norms for 3 vectors
        ptr = [0, 3, 6, 9]     # each norm maps to 3 components
        
    Args:
        values: The actual values
        ptr: Pointers to where each value's components begin/end
        dtype: The dtype for values (ptr will always be torch.long)
    """
    values: torch.Tensor
    ptr: torch.Tensor
    dtype: torch.dtype = torch.float32

    @classmethod
    def empty(cls, size: int, dtype: torch.dtype = torch.float32):
        """Create an empty index with just start and end pointers."""
        return cls(
            values=torch.tensor([]),
            ptr=torch.tensor([0, size], dtype=torch.long),
            dtype=dtype
        )

    def __post_init__(self):
        if not isinstance(self.values, torch.Tensor):
            self.values = torch.tensor(self.values, dtype=self.dtype)
        else:
            self.values = self.values.to(dtype=self.dtype)
            
        if not isinstance(self.ptr, torch.Tensor):
            self.ptr = torch.tensor(self.ptr, dtype=torch.long)
        else:
            self.ptr = self.ptr.to(dtype=torch.long)
            
    def __len__(self):
        return len(self.values)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 3  # Small batch for readable output
    
    # Create test tensors with known values
    vectors = torch.tensor([
        [2.0, 0.0, 1.0],  # norm ≈ 2.236
        [2.0, 0.0, 0.0],  # norm = 2.0
        [1.0, 1.0, 1.0],  # norm ≈ 1.732
    ])
    
    symmetric = torch.tensor([
        [[1.0, 0.5, 0.0],
         [0.5, 1.0, 0.0],
         [0.0, 0.0, 1.0]],  # Symmetric with clear pattern
        [[2.0, 0.0, 0.0],
         [0.0, 2.0, 0.0],
         [0.0, 0.0, 2.0]],  # Diagonal
        [[0.5, 0.0, 0.0],
         [0.0, 0.5, 0.0],
         [0.0, 0.0, 0.5]]   # Scaled diagonal
    ])
    
    skew = torch.tensor([
        [[0.0, 1.0, 0.0],
         [-1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]],  # Simple rotation in xy
        [[0.0, 0.0, 2.0],
         [0.0, 0.0, 0.0],
         [-2.0, 0.0, 0.0]], # Rotation in xz
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.5],
         [0.0, -0.5, 0.0]]  # Rotation in yz
    ])
    
    # Create TensorData objects
    vec_tensor = TensorData(vectors)
    sym_tensor = TensorData(symmetric, symmetry=1)
    skew_tensor = TensorData(skew, symmetry=-1)
    combined = TensorData.cat([sym_tensor, skew_tensor, vec_tensor])
    
    print("\nBasic combined tensor structure:")
    print(f"Shape: {combined.shape}")
    print(f"Field ptr: {combined.ptr.ptr}")
    print(f"Ranks: {combined.rank.values}")
    print(f"Symmetries: {combined.symmetry.values}")
    
    if False:  # Original tests
        print("\nDetailed tensor tests:")
        tensor_types = {
            "General rank-2": symmetric,  # Using symmetric as general rank-2
            "Symmetric": symmetric,
            "Skew-symmetric": skew,
            "Vector": vectors,
        }
        
        for name, tensor in tensor_types.items():
            print(f"\n{name} tensor:")
            td = TensorData(tensor)
            print(f"Rank: {td.rank.values[0]}")
            print(f"Symmetry: {td.symmetry.values[0]}")
            print(f"Shape after flattening: {td.shape}")
            print(f"Field ptr: {td.ptr.ptr}")
            norms = td.norms
            print(f"Norms shape: {norms.values.shape}")
            print(f"Norm ptr: {norms.ptr}")
    
    if True:  # Test irrep conversion
        print("\nTesting irrep conversion:")
        # Convert to irreps
        irrep_tensor, irrep_str = combined.to_irreps()
        
        print("\nIrrep structure:")
        print(f"Original tensor shape: {combined.shape}")
        print(f"Irrep tensor shape: {irrep_tensor.shape}")
        print(f"Irrep string: {irrep_str}")
        
        # Test individual fields
        for i in range(combined.num_fields):
            field = combined.get_field(i)
            field_irrep, field_str = field.to_irreps()
            print(f"\nField {i}:")
            print(f"Type: {'Symmetric' if field.symmetry.values[0] == 1 else 'Skew-symmetric' if field.symmetry.values[0] == -1 else 'Vector'}")
            print(f"Original shape: {field.shape}")
            print(f"Irrep shape: {field_irrep.shape}")
            print(f"Irrep string: {field_str}")