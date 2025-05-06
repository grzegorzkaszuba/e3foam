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
import time
from adapters import IrrepAdapter, CartesianAdapter
# Try relative import first (for when e3foam is a package)
try:
    from .utils import project_tensor_to_2d
# If that fails, try absolute import (for when e3foam is in path)
except ImportError:
    try:
        from e3foam.tensors.utils import project_tensor_to_2d
    # If both fail, provide a helpful error message
    except ImportError:
        raise ImportError(
            "Could not import project_tensor_to_2d. Make sure either:\n"
            "1. e3foam is installed as a package, or\n"
            "2. The e3foam directory is in your Python path."
        )


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
    
    def __init__(self, tensor: torch.Tensor,
                 rank: int = None,
                 symmetry: int = 0,
                 is_multi_field: bool = False,
                 is_flattened: bool = False):
        """
        Initialize TensorData with a tensor and optional properties.
        
        Args:
            tensor: Input tensor
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

        if inferred_rank == 0 and self.tensor.ndim == 1:
            self.tensor = self.tensor.reshape(-1, 1)

        
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
        elif values_per_field in [9, 6]:
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
        
        # Create new TensorData with combined properties
        result = cls(cat_tensor, is_multi_field=True)
        result.ptr = TensorIndex(torch.tensor([]), ptr, dtype=torch.long)
        result.rank = TensorIndex(ranks, ptr, dtype=torch.long)
        result.symmetry = TensorIndex(symmetries, ptr, dtype=torch.long)
        
        return result

    @classmethod
    def append(cls, tensordatas: List['TensorData']) -> 'TensorData':
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
        first = tensordatas[0]
        for n, td in enumerate(tensordatas[1:]):
            # Validate structural properties match exactly
            if not torch.equal(first.ptr.ptr, td.ptr.ptr):
                raise ValueError(f"Field pointers must match exactly, mismatch on index 0 and {n+1}")

            if not torch.equal(first.rank.values, td.rank.values):
                raise ValueError(f"Rank values must match exactly, mismatch on index 0 and {n+1}")

            if not torch.equal(first.symmetry.values, td.symmetry.values):
                raise ValueError(f"Symmetry values must match exactly, mismatch on index 0 and {n+1}")
        
        # If all validations pass, combine the tensors
        combined_tensor = torch.cat([td.tensor for td in tensordatas], dim=0)
        
        # Create new TensorData with same properties
        result = TensorData(combined_tensor, is_multi_field=True)
        result.ptr = first.ptr
        result.rank = first.rank
        result.symmetry = first.symmetry
        
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
        
        return result

    def to_irreps(self):
        """
        Convert tensor data to irreducible representation.
        
        # TODO: Refactor this method to have a cleaner API:
        #  1. Rename this method to something like `to_irreps_full` that returns all 3 values
        #  2. Create a simpler `to_irreps` method that only returns the irrep tensor
        #  3. Consider making the irreps_str and cartesian_str properties of the class
        
        Returns:
            tuple: (irrep_tensor, irreps_str, cartesian_str)
                - irrep_tensor: Tensor in irreducible representation
                - irreps_str: String representation of irreps
                - cartesian_str: String representation of Cartesian fields
        """
        irrep_tensors = []
        irrep_strs = []
        cartesian_parts = []
        
        for i in range(self.num_fields):
            field = self.get_field(i)
            rank = field.rank.values[0].item()
            symmetry = field.symmetry.values[0].item()
            size = field.tensor.shape[1]
            
            # Determine field type and CartesianTensor signature in one step
            if rank == 0:
                field_type = "scalar"
                ct_signature = "0"
                irrep_str = "1x0e"
                # Scalar field - just use the tensor directly
                irrep_data = field.tensor
                
            elif rank == 1:
                field_type = "vector"
                ct_signature = "i"
                irrep_str = "1x1o"
                # Vector field - just use the tensor directly
                irrep_data = field.tensor
                
            elif rank == 2:
                # Create full tensor with same device/dtype as input
                batch_size = field.tensor.shape[0]
                full_tensor = torch.zeros(
                    (batch_size, 3, 3), 
                    device=field.tensor.device, 
                    dtype=field.tensor.dtype
                )
                
                if symmetry == 0:  # No symmetry
                    field_type = "tensor"
                    ct_signature = "ij"
                    full_tensor = field.tensor.reshape(batch_size, 3, 3)
                    
                elif symmetry == 1:  # Symmetric
                    field_type = "symmetric"
                    ct_signature = "ij=ji"
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
                    
                else:  # symmetry == -1, Skew-symmetric
                    field_type = "skew"
                    ct_signature = "ij=-ji"
                    # [xy, xz, yz] -> 3x3
                    full_tensor[:, 0, 1] = field.tensor[:, 0]   # xy
                    full_tensor[:, 1, 0] = -field.tensor[:, 0]  # -xy
                    full_tensor[:, 0, 2] = field.tensor[:, 1]   # xz
                    full_tensor[:, 2, 0] = -field.tensor[:, 1]  # -xz
                    full_tensor[:, 1, 2] = field.tensor[:, 2]   # yz
                    full_tensor[:, 2, 1] = -field.tensor[:, 2]  # -yz
                    
                # Convert to irreps using CartesianTensor
                ct = CartesianTensor(ct_signature)
                irrep_data = ct.from_cartesian(full_tensor)
                irrep_str = str(o3.Irreps(ct))
            
            # Add to our lists
            cartesian_parts.append(f"{field_type}[{size}]:{ct_signature}")
            irrep_tensors.append(irrep_data)
            irrep_strs.append(irrep_str)
        
        # Combine tensors and strings
        combined_tensor = torch.cat(irrep_tensors, dim=-1)
        irreps_str = str(o3.Irreps(" + ".join(irrep_strs)))
        cartesian_str = "+".join(cartesian_parts)
        
        return combined_tensor, irreps_str, cartesian_str

    @classmethod
    def from_irreps(cls, irrep_tensor: torch.Tensor, irreps_str: str, cartesian_str: str, 
                    cache_rtps: bool = False, cached_rtps: dict = None) -> 'TensorData':
        """
        Create TensorData from irrep tensor and strings.
        
        Args:
            irrep_tensor: Tensor in irrep basis from e3nn
            irreps_str: String describing the irreps structure
            cartesian_str: String describing the original tensor structure
            cache_rtps: If True, create and cache reduced tensor products for future use
            cached_rtps: Dictionary of pre-computed reduced tensor products to use
            
        Returns:
            TensorData object with proper structure and symmetry
        """
        # TODO: Implement caching for CartesianTensor operations
        # The CartesianTensor class can be slow for repeated operations.
        # Consider caching the results of reducedtensorproducts to improve performance,
        # especially when processing batches of similar tensor structures.
        
        # Parse the irreps string
        irreps = o3.Irreps(irreps_str)
        
        # Parse the cartesian string to get field information
        cartesian_parts = cartesian_str.split("+")
        
        # Map irreps to their positions for quick lookup
        # We'll use just the l value as the key since parity might differ
        irrep_map = {}
        start_idx = 0
        for i, (mul, ir) in enumerate(irreps):
            irrep_size = mul * (2 * ir.l + 1)
            key = ir.l  # Use only the l value as key
            if key not in irrep_map:
                irrep_map[key] = []
            irrep_map[key].append((i, start_idx, start_idx + irrep_size, ir.p))  # Store parity too
            start_idx += irrep_size
        
        # Create a copy to track used irreps
        used_irreps = {key: [] for key in irrep_map}
        
        # Store individual field tensors and their properties
        field_tensors = []
        field_ranks = []
        field_symmetries = []
        
        # Initialize cache for CartesianTensor objects and their RTPs
        ct_cache = {}
        rtp_cache = {}
        
        # Process each field according to cartesian string
        for part in cartesian_parts:
            # Parse the field type, size, and CartesianTensor signature
            field_info, ct_signature = part.split(":")
            field_type, size_str = field_info.split("[")
            size = int(size_str.rstrip("]"))
            
            # Determine field properties
            batch_size = irrep_tensor.shape[0]
            
            if field_type == "scalar":
                rank = 0
                symmetry = 0
                
                # Get scalar irrep (l=0)
                key = 0  # l=0 for scalar
                if key in irrep_map and len(irrep_map[key]) > len(used_irreps[key]):
                    idx = len(used_irreps[key])
                    _, start, end, _ = irrep_map[key][idx]
                    used_irreps[key].append(idx)  # Mark as used
                    field_data = irrep_tensor[:, start:end]
                else:
                    raise ValueError(f"No scalar irrep (l=0) found for field: {part}")
                
                # Reshape to match expected format
                field_tensor = field_data.reshape(batch_size, 1)
                
            elif field_type == "vector":
                rank = 1
                symmetry = 0
                
                # Get vector irrep (l=1)
                key = 1  # l=1 for vector
                if key in irrep_map and len(irrep_map[key]) > len(used_irreps[key]):
                    idx = len(used_irreps[key])
                    _, start, end, _ = irrep_map[key][idx]
                    used_irreps[key].append(idx)  # Mark as used
                    field_data = irrep_tensor[:, start:end]
                else:
                    raise ValueError(f"No vector irrep (l=1) found for field: {part}")
                
                # Reshape to match expected format
                field_tensor = field_data.reshape(batch_size, 3)
                
            elif field_type in ["tensor", "symmetric", "skew"]:
                rank = 2
                
                # Set symmetry based on field type
                if field_type == "tensor":
                    symmetry = 0
                elif field_type == "symmetric":
                    symmetry = 1
                else:  # skew
                    symmetry = -1
                
                # Get CartesianTensor for this field type
                if ct_signature in ct_cache:
                    ct = ct_cache[ct_signature]
                else:
                    ct = CartesianTensor(ct_signature)
                    if cache_rtps:
                        ct_cache[ct_signature] = ct
                
                # Extract irreps for this field
                field_irreps = []
                
                # Determine which irreps to extract based on field type
                if field_type == "tensor":
                    # General tensor: l=0, l=1, l=2
                    irrep_types = [0, 1, 2]  # Just l values
                elif field_type == "symmetric":
                    # Symmetric tensor: l=0, l=2
                    irrep_types = [0, 2]
                else:  # skew
                    # Skew-symmetric tensor: l=1
                    irrep_types = [1]
                
                # Extract each irrep type
                for l in irrep_types:
                    key = l
                    if key in irrep_map and len(irrep_map[key]) > len(used_irreps[key]):
                        idx = len(used_irreps[key])
                        _, start, end, _ = irrep_map[key][idx]
                        used_irreps[key].append(idx)  # Mark as used
                        field_irreps.append(irrep_tensor[:, start:end])
                    else:
                        raise ValueError(f"No irrep (l={l}) found for field: {part}")
                
                # Concatenate irreps
                field_data = torch.cat(field_irreps, dim=1)
                
                # Convert to Cartesian tensor
                # Use cached RTPs if available
                if cached_rtps and ct_signature in cached_rtps:
                    rtp = cached_rtps[ct_signature]
                    field_tensor_3d = ct.to_cartesian(field_data, rtp=rtp)
                else:
                    # Generate RTP if caching is enabled
                    if cache_rtps:
                        if ct_signature not in rtp_cache:
                            rtp_cache[ct_signature] = ct.reduced_tensor_products(field_data)
                        field_tensor_3d = ct.to_cartesian(field_data, rtp=rtp_cache[ct_signature])
                    else:
                        field_tensor_3d = ct.to_cartesian(field_data)
                
                # Flatten tensor according to symmetry
                if symmetry == 0:  # No symmetry
                    field_tensor = field_tensor_3d.reshape(batch_size, 9)
                elif symmetry == 1:  # Symmetric
                    # Extract unique components [xx, xy, xz, yy, yz, zz]
                    field_tensor = torch.zeros((batch_size, 6), device=field_tensor_3d.device, dtype=field_tensor_3d.dtype)
                    field_tensor[:, 0] = field_tensor_3d[:, 0, 0]  # xx
                    field_tensor[:, 1] = field_tensor_3d[:, 0, 1]  # xy
                    field_tensor[:, 2] = field_tensor_3d[:, 0, 2]  # xz
                    field_tensor[:, 3] = field_tensor_3d[:, 1, 1]  # yy
                    field_tensor[:, 4] = field_tensor_3d[:, 1, 2]  # yz
                    field_tensor[:, 5] = field_tensor_3d[:, 2, 2]  # zz
                else:  # symmetry == -1, Skew-symmetric
                    # Extract unique components [xy, xz, yz]
                    field_tensor = torch.zeros((batch_size, 3), device=field_tensor_3d.device, dtype=field_tensor_3d.dtype)
                    field_tensor[:, 0] = field_tensor_3d[:, 0, 1]  # xy
                    field_tensor[:, 1] = field_tensor_3d[:, 0, 2]  # xz
                    field_tensor[:, 2] = field_tensor_3d[:, 1, 2]  # yz
            else:
                raise ValueError(f"Unknown field type: {field_type}")
            
            # Store field data and properties
            field_tensors.append(field_tensor)
            field_ranks.append(rank)
            field_symmetries.append(symmetry)
        
        # Return cached RTPs if requested
        if cache_rtps:
            return cls.from_fields(field_tensors, field_ranks, field_symmetries), rtp_cache
        
        # Create TensorData from fields
        return cls.from_fields(field_tensors, field_ranks, field_symmetries)

    @classmethod
    def from_fields(cls, field_tensors, field_ranks, field_symmetries):
        """
        Create TensorData from individual field tensors and properties.
        
        Args:
            field_tensors: List of tensors for each field
            field_ranks: List of ranks for each field
            field_symmetries: List of symmetry values for each field
            
        Returns:
            TensorData object with proper structure
        """
        # Create individual TensorData objects
        tensor_datas = []
        for i in range(len(field_tensors)):
            td = cls(
                field_tensors[i],
                rank=field_ranks[i],
                symmetry=field_symmetries[i],
                is_flattened=True
            )
            tensor_datas.append(td)
        
        return cls.cat(tensor_datas)

    def project_to_2d(self) -> 'TensorData':
        """
        Project a 3D TensorData to 2D by zeroing out all components involving the z-dimension.
        
        Returns:
            A new TensorData with z-components set to zero
        
        Note:
            This preserves the tensor structure but zeros out any component that involves
            the z-dimension (index 2 in 0-based indexing).
        """
        # Create a copy of the tensor
        projected = self.tensor.clone()
        
        # Process each field
        for i in range(self.num_fields):
            field = self.get_field(i)
            rank = field.rank.values[0].item()
            symmetry = field.symmetry.values[0].item()
            start, end = self.ptr.ptr[i], self.ptr.ptr[i+1]
            
            if rank == 0:
                # Scalars are unchanged in projection
                pass
            
            elif rank == 1:
                # For vectors, zero out the z-component (index 2)
                projected[:, start+2] = 0.0
            
            elif rank == 2:
                if symmetry == 0:  # General tensor
                    # Zero out any component involving z (index 2)
                    # In flattened form: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
                    z_components = [2, 5, 6, 7, 8]  # xz, yz, zx, zy, zz
                    for idx in z_components:
                        projected[:, start+idx] = 0.0
                
                elif symmetry == 1:  # Symmetric tensor
                    # In flattened form: [xx, xy, xz, yy, yz, zz]
                    z_components = [2, 4, 5]  # xz, yz, zz
                    for idx in z_components:
                        projected[:, start+idx] = 0.0
                
                elif symmetry == -1:  # Skew-symmetric tensor
                    # In flattened form: [xy, xz, yz]
                    z_components = [1, 2]  # xz, yz
                    for idx in z_components:
                        projected[:, start+idx] = 0.0
        
        # TODO: manage trace management
        # Consider how to handle trace redistribution for traceless tensors
        # when projecting to 2D
        
        # Create a new TensorData with the projected tensor
        result = TensorData(projected, is_multi_field=True)
        result.ptr = self.ptr
        result.rank = self.rank
        result.symmetry = self.symmetry
        
        return result

    def get_irrep_adapter(self):
        """
        Creates an optimized object that converts tensor data to irrep representation.

        Returns:
            IrrepAdapter: An object that transforms tensor data to irrep representation
        """
        # Create a adapter object instead of a function
        return IrrepAdapter(self)

    def get_cartesian_adapter(self, cache_rtps=False, project_2d=False,
                           projection_plane='xy', preserve_trace=True):
        """Get an adapter for converting to Cartesian representation."""
        return CartesianAdapter(self, cache_rtps=cache_rtps, project_2d=project_2d,
                             projection_plane=projection_plane, preserve_trace=preserve_trace)

    def get_irrep_adapter(self):
        """Get an adapter for converting to irrep representation."""
        return IrrepAdapter(self)

    def to(self, device=None, dtype=None):
        """
        Move the tensor data to the specified device and data type.

        Args:
            device: The device to move tensors to
            dtype: The data type to convert tensors to

        Returns:
            self for method chaining
        """
        if device is not None or dtype is not None:
            self.tensor = self.tensor.to(device=device, dtype=dtype)

            # Also move indices to the same device
            if device is not None:
                if hasattr(self, 'ptr'):
                    self.ptr.to(device=device)
                if hasattr(self, 'rank'):
                    self.rank.to(device=device)
                if hasattr(self, 'symmetry'):
                    self.symmetry.to(device=device)

        return self

    def __getitem__(self, idx):
        """
        Slice the TensorData along the batch dimension.

        Args:
            idx: Index or slice to select from the batch dimension

        Returns:
            A new TensorData object with the selected batch items
        """
        # Create a copy of the current object
        import copy
        result = copy.deepcopy(self)

        # Slice the tensor along the batch dimension
        # For single indices, preserve the batch dimension by adding a new axis
        if isinstance(idx, int):
            result.tensor = self.tensor[idx:idx+1]  # Keep as (1, features) instead of (features,)
        else:
            result.tensor = self.tensor[idx]

        return result




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

    def to(self, device=None, dtype=None):
        """
        Move the tensor index to the specified device and data type.

        Args:
            device: The device to move tensors to
            dtype: The data type to convert tensors to

        Returns:
            self for method chaining
        """
        if device is not None:
            self.ptr = self.ptr.to(device=device)

        if device is not None or dtype is not None:
            if self.values is not None:
                self.values = self.values.to(device=device, dtype=dtype)

        return self


