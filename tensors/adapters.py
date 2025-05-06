from dataclasses import dataclass
import copy
import torch
from typing import List
import numpy as np
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

class IrrepAdapter:
    """
    Object that efficiently converts tensor data to irrep representation.
    """
    def __init__(self, tensor_data):
        """
        Initialize the IrrepAdapter with a TensorData object.

        Args:
            tensor_data: TensorData object to base the transformation on
        """
        # Store original tensor data properties
        self.tensor_data = tensor_data
        self.num_fields = tensor_data.num_fields
        self.field_transformers = []

        # Get the irreps string and cartesian string
        _, self.irreps_str, self.cartesian_str = tensor_data.to_irreps()

    def __call__(self, tensor):
        """
        Transform tensor data to irrep representation.

        Args:
            tensor: Tensor data to transform (can be TensorData, torch.Tensor, or numpy array)

        Returns:
            tuple: (irrep_tensor, irreps_str)
        """
        from e3foam.tensors.base import TensorData  # Import here to avoid circular imports
        
        # Case a: Input is already a TensorData object
        if isinstance(tensor, TensorData):
            temp_td = tensor
        # Case b: Input is a torch.Tensor or numpy array
        elif isinstance(tensor, torch.Tensor) or isinstance(tensor, np.ndarray):
            # Convert numpy array to torch tensor if needed
            if isinstance(tensor, np.ndarray):
                tensor = torch.tensor(tensor)
            
            # Case b2: Check if our template has multiple fields
            if self.tensor_data.num_fields > 1:
                raise ValueError(
                    "Cannot automatically convert raw tensor to multiple fields. "
                    "Please convert to TensorData first with appropriate field structure."
                )
            
            # Case b1: Single field - use the properties from our template
            rank = self.tensor_data.rank.values[0].item()
            symmetry = self.tensor_data.symmetry.values[0].item()
            
            # Create a proper TensorData object with inferred structure
            temp_td = TensorData(tensor, rank=rank, symmetry=symmetry)
        # Case c: Unrecognized input type
        else:
            raise TypeError(
                f"Unsupported input type: {type(tensor)}. "
                "Input must be a TensorData object, torch.Tensor, or numpy.ndarray."
            )
        
        # Use the existing to_irreps method for consistency
        irrep_tensor, irreps_str, _ = temp_td.to_irreps()
        
        # Return the irrep tensor and string (matching original behavior)
        return irrep_tensor, irreps_str

    def to(self, device=None, dtype=None):
        """
        Move the adapter's internal tensors to the specified device and data type.

        Args:
            device: The device to move tensors to
            dtype: The data type to convert tensors to

        Returns:
            self for method chaining
        """
        # Move any cached tensors to the specified device and dtype
        # Currently IrrepAdapter doesn't cache tensors, but adding for future-proofing
        return self


class CartesianAdapter:  # was CartesianAdapter

    def __init__(self, tensor_data, cache_rtps=False, project_2d=False, projection_plane='xy', preserve_trace=True):
        """
        Initialize the CartesianAdapter with a TensorData object.

        Args:
            tensor_data: TensorData object to base the transformation on
            cache_rtps: If True, pre-compute reduced tensor products for better performance
            project_2d: If True, project the result to 2D
            projection_plane: Plane to project to ('xy', 'xz', or 'yz')
            preserve_trace: Whether to preserve the trace during projection
        """
        # Store original tensor data properties
        self.tensor_data = tensor_data

        # Get irrep and Cartesian information
        _, self.irreps_str, self.cartesian_str = tensor_data.to_irreps()
        self.irreps = o3.Irreps(self.irreps_str)
        self.cartesian_parts = self.cartesian_str.split("+")

        # Store projection settings
        self.project_2d = project_2d
        self.projection_plane = projection_plane
        self.preserve_trace = preserve_trace

        # Map irreps to their positions for quick lookup
        self.irrep_map = {}
        start_idx = 0
        for i, (mul, ir) in enumerate(self.irreps):
            irrep_size = mul * (2 * ir.l + 1)
            key = ir.l  # Use only the l value as key
            if key not in self.irrep_map:
                self.irrep_map[key] = []
            self.irrep_map[key].append((i, start_idx, start_idx + irrep_size, ir.p))
            start_idx += irrep_size

        # Create a copy to track used irreps
        self.used_irreps = {key: [] for key in self.irrep_map}

        # Cache for CartesianTensor objects and RTPs
        self.ct_cache = {}
        self.rtp_cache = {}
        self.cache_rtps = cache_rtps

        # Prepare transformers for each field
        self.field_transformers = []

        # Process each field
        for part in self.cartesian_parts:
            # Parse the field type, size, and CartesianTensor signature
            field_info, ct_signature = part.split(":")
            field_type, size_str = field_info.split("[")
            size = int(size_str.rstrip("]"))

            if field_type == "scalar":
                # Scalar field - identity transformation
                key = 0
                idx = len(self.used_irreps[key])
                if idx >= len(self.irrep_map.get(key, [])):
                    raise ValueError(f"No irrep (l=0) found for field: {part}")
                _, start, end, _ = self.irrep_map[key][idx]
                self.used_irreps[key].append(idx)

                self.field_transformers.append(self._make_scalar_transformer(start, end))

            elif field_type == "vector":
                # Vector field - identity transformation
                key = 1
                idx = len(self.used_irreps[key])
                if idx >= len(self.irrep_map.get(key, [])):
                    raise ValueError(f"No irrep (l=1) found for field: {part}")
                _, start, end, _ = self.irrep_map[key][idx]
                self.used_irreps[key].append(idx)

                self.field_transformers.append(self._make_vector_transformer(start, end))

            elif field_type in ["tensor", "symmetric", "skew"]:
                # Get CartesianTensor for this field type
                if ct_signature in self.ct_cache:
                    ct = self.ct_cache[ct_signature]
                else:
                    ct = CartesianTensor(ct_signature)
                    self.ct_cache[ct_signature] = ct

                # Extract irreps for this field
                field_irreps_info = []

                # Determine which irreps to extract based on field type
                if field_type == "tensor":
                    irrep_types = [0, 1, 2]
                elif field_type == "symmetric":
                    irrep_types = [0, 2]
                else:  # skew
                    irrep_types = [1]

                # Extract each irrep type
                for l in irrep_types:
                    key = l
                    if key not in self.irrep_map or len(self.irrep_map[key]) <= len(self.used_irreps.get(key, [])):
                        raise ValueError(f"No irrep (l={l}) found for field: {part}")
                    idx = len(self.used_irreps.get(key, []))
                    _, start, end, p = self.irrep_map[key][idx]
                    if key not in self.used_irreps:
                        self.used_irreps[key] = []
                    self.used_irreps[key].append(idx)
                    field_irreps_info.append((start, end))

                # Create the appropriate transformer
                if field_type == "tensor":
                    self.field_transformers.append(
                        self._make_tensor_transformer(field_irreps_info, ct, ct_signature)
                    )
                elif field_type == "symmetric":
                    self.field_transformers.append(
                        self._make_symmetric_transformer(field_irreps_info, ct, ct_signature)
                    )
                else:  # skew-symmetric
                    self.field_transformers.append(
                        self._make_skew_transformer(field_irreps_info, ct, ct_signature)
                    )
            else:
                raise ValueError(f"Unknown field type: {field_type}")

    def _make_scalar_transformer(self, start, end):
        """Create a transformer for scalar fields."""
        def transform(irrep_tensor):
            return irrep_tensor[:, start:end]
        return transform

    def _make_vector_transformer(self, start, end):
        """Create a transformer for vector fields."""
        def transform(irrep_tensor):
            return irrep_tensor[:, start:end]
        return transform

    def _make_tensor_transformer(self, irrep_info, ct, signature):
        """Create a transformer for general tensor fields."""
        def transform(irrep_tensor):
            # Concatenate the irreps for this field
            field_data = torch.cat([
                irrep_tensor[:, start:end] for start, end in irrep_info
            ], dim=1)

            # Convert to Cartesian tensor
            if signature in self.rtp_cache:
                field_tensor_3d = ct.to_cartesian(field_data, rtp=self.rtp_cache[signature])
            else:
                if self.cache_rtps:
                    self.rtp_cache[signature] = ct.reduced_tensor_products(field_data)
                    field_tensor_3d = ct.to_cartesian(field_data, rtp=self.rtp_cache[signature])
                else:
                    field_tensor_3d = ct.to_cartesian(field_data)

            # Flatten to 9 components
            return field_tensor_3d.reshape(field_tensor_3d.shape[0], 9)

        return transform

    def _make_symmetric_transformer(self, irrep_info, ct, signature):
        """Create a transformer for symmetric tensor fields."""
        def transform(irrep_tensor):
            # Concatenate the irreps for this field
            field_data = torch.cat([
                irrep_tensor[:, start:end] for start, end in irrep_info
            ], dim=1)

            # Convert to Cartesian tensor
            if signature in self.rtp_cache:
                field_tensor_3d = ct.to_cartesian(field_data, rtp=self.rtp_cache[signature])
            else:
                if self.cache_rtps:
                    self.rtp_cache[signature] = ct.reduced_tensor_products(field_data)
                    field_tensor_3d = ct.to_cartesian(field_data, rtp=self.rtp_cache[signature])
                else:
                    field_tensor_3d = ct.to_cartesian(field_data)

            # Extract unique components [xx, xy, xz, yy, yz, zz]
            batch_size = field_tensor_3d.shape[0]
            field_tensor = torch.zeros((batch_size, 6), device=field_tensor_3d.device, dtype=field_tensor_3d.dtype)
            field_tensor[:, 0] = field_tensor_3d[:, 0, 0]  # xx
            field_tensor[:, 1] = field_tensor_3d[:, 0, 1]  # xy
            field_tensor[:, 2] = field_tensor_3d[:, 0, 2]  # xz
            field_tensor[:, 3] = field_tensor_3d[:, 1, 1]  # yy
            field_tensor[:, 4] = field_tensor_3d[:, 1, 2]  # yz
            field_tensor[:, 5] = field_tensor_3d[:, 2, 2]  # zz

            return field_tensor

        return transform

    def _make_skew_transformer(self, irrep_info, ct, signature):
        """Create a transformer for skew-symmetric tensor fields."""
        def transform(irrep_tensor):
            # Concatenate the irreps for this field
            field_data = torch.cat([
                irrep_tensor[:, start:end] for start, end in irrep_info
            ], dim=1)

            # Convert to Cartesian tensor
            if signature in self.rtp_cache:
                field_tensor_3d = ct.to_cartesian(field_data, rtp=self.rtp_cache[signature])
            else:
                if self.cache_rtps:
                    self.rtp_cache[signature] = ct.reduced_tensor_products(field_data)
                    field_tensor_3d = ct.to_cartesian(field_data, rtp=self.rtp_cache[signature])
                else:
                    field_tensor_3d = ct.to_cartesian(field_data)

            # Extract unique components [xy, xz, yz]
            batch_size = field_tensor_3d.shape[0]
            field_tensor = torch.zeros((batch_size, 3), device=field_tensor_3d.device, dtype=field_tensor_3d.dtype)
            field_tensor[:, 0] = field_tensor_3d[:, 0, 1]  # xy
            field_tensor[:, 1] = field_tensor_3d[:, 0, 2]  # xz
            field_tensor[:, 2] = field_tensor_3d[:, 1, 2]  # yz

            return field_tensor

        return transform

    def __call__(self, irrep_tensor):
        """
        Transform irrep tensor to Cartesian representation.

        Args:
            irrep_tensor: Tensor in irrep representation

        Returns:
            torch.Tensor: Tensor in Cartesian representation
        """
        # Apply each field transformer
        field_tensors = [
            transformer(irrep_tensor) for transformer in self.field_transformers
        ]

        # Concatenate the results
        cartesian_tensor = torch.cat(field_tensors, dim=1)

        # Apply 2D projection if requested
        if self.project_2d:
            cartesian_tensor = project_tensor_to_2d(
                cartesian_tensor,
                plane=self.projection_plane,
                preserve_trace=self.preserve_trace
            )

        return cartesian_tensor

    def set_rtps(self, rtps):
        """
        Set pre-computed RTPs for this adapter.

        Args:
            rtps: Dictionary of pre-computed RTPs
        """
        self.rtp_cache = rtps

    def get_rtps(self):
        """
        Get the current RTP cache.

        Returns:
            dict: The current RTP cache
        """
        return self.rtp_cache

    def project_to_2d(self, irrep_tensor, plane=None, preserve_trace=None):
        """
        Transform irrep tensor to Cartesian and project to 2D.

        Args:
            irrep_tensor: Tensor in irrep representation
            plane: Projection plane ('xy', 'xz', or 'yz'), defaults to self.projection_plane
            preserve_trace: Whether to preserve the trace, defaults to self.preserve_trace

        Returns:
            torch.Tensor: Projected tensor in Cartesian representation
        """
        # Use instance defaults if not specified
        plane = plane or self.projection_plane
        preserve_trace = preserve_trace if preserve_trace is not None else self.preserve_trace

        # Transform to Cartesian
        cartesian_tensor = self(irrep_tensor)

        # Project to 2D
        return project_tensor_to_2d(cartesian_tensor, plane=plane, preserve_trace=preserve_trace)

    def to(self, device=None, dtype=None):
        """
        Move the adapter's internal tensors to the specified device and data type.

        Args:
            device: The device to move tensors to
            dtype: The data type to convert tensors to

        Returns:
            self for method chaining
        """
        # Move cached RTPs to the specified device and dtype
        if self.rtp_cache:
            for key, rtp in self.rtp_cache.items():
                if hasattr(rtp, 'to'):
                    self.rtp_cache[key] = rtp.to(device=device, dtype=dtype)

        return self