import os
import shutil
import torch
from typing import Union, Dict
from tensors.base import TensorData

def format_float(value: float) -> str:
    """Format float values according to OpenFOAM conventions."""
    sci_notation_threshold = 1e-04

    if abs(value) < sci_notation_threshold:
        if abs(value) < 1e-15:
            return "0"
        formatted = f"{value:.5e}"
        return formatted.replace("e+", "e")
    return f"{value:.9f}"

def format_tensor_field(tensor_data: TensorData, field_idx: int = 0) -> torch.Tensor:
    """Format TensorData for OpenFOAM field file."""
    field = tensor_data.get_field(field_idx) if len(tensor_data.ptr.ptr) > 2 else tensor_data
    rank = field.rank.values[0].item()
    symmetry = field.symmetry.values[0].item()

    if rank == 0:  # Scalar
        return field.tensor
    elif rank == 1:  # Vector
        return field.tensor  # Already [batch_size, 3]
    elif rank == 2:  # Rank-2 tensor
        if symmetry == 1:  # Symmetric - expand from [xx, xy, xz, yy, yz, zz]
            batch_size = field.tensor.shape[0]
            full = torch.zeros((batch_size, 6), dtype=field.tensor.dtype)
            full[:, [0, 1, 2, 3, 4, 5]] = field.tensor  # [xx, xy, xz, yy, yz, zz]
            return full
        elif symmetry == -1:  # Skew-symmetric - expand from [xy, xz, yz]
            batch_size = field.tensor.shape[0]
            full = torch.zeros((batch_size, 6), dtype=field.tensor.dtype)
            full[:, [1, 2, 4]] = field.tensor  # [xy, xz, yz]
            full[:, [0, 3, 5]] = 0  # Diagonal is zero
            return full
        else:  # No symmetry - reshape to OpenFOAM format
            return field.tensor.reshape(-1, 9)

def inject_tensor_field(foamfile: str, tensor_data: TensorData, field_idx: int = 0):
    """
    Inject TensorData into OpenFOAM field file.

    Args:
        foamfile: Path to OpenFOAM field file
        tensor_data: TensorData object
        field_idx: Which field to inject if multi-field tensor
    """
    # Format tensor for OpenFOAM
    formatted_tensor = format_tensor_field(tensor_data, field_idx)

    # Read file
    with open(foamfile, 'r') as file:
        lines = file.readlines()

    # Find internalField section
    start_idx = next(i for i, line in enumerate(lines) if 'internalField' in line)
    end_idx = next(i for i, line in enumerate(lines[start_idx:]) if line.strip() == ')') + start_idx

    # Write file with new data
    with open(foamfile, 'w') as file:
        # Write header
        file.writelines(lines[:start_idx + 3])

        # Write formatted data
        for row in formatted_tensor:
            if len(row) == 1:  # Scalar
                file.write(f"{format_float(row.item())}\n")
            else:  # Vector or tensor
                formatted_row = ' '.join(format_float(x) for x in row)
                file.write(f"({formatted_row})\n")

        # Write footer
        file.writelines(lines[end_idx:])

def inject_tensor_fields(tensor_data: TensorData, template_dir: str, output_dir: str):
    """
    Inject multi-field TensorData into multiple OpenFOAM files.

    Args:
        tensor_data: TensorData object with multiple fields
        template_dir: Directory with template files
        output_dir: Directory to write output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Process each field
    for i in range(len(tensor_data.ptr.ptr) - 1):
        field = tensor_data.get_field(i)
        field_type = {
            (0, 0): "scalar",
            (1, 0): "vector",
            (2, 0): "tensor",
            (2, 1): "symmTensor",
            (2, -1): "skewTensor"
        }[(field.rank.values[0].item(), field.symmetry.values[0].item())]

        # Copy template and inject data
        template_file = os.path.join(template_dir, f"{field_type}.template")
        output_file = os.path.join(output_dir, f"field_{i}")
        shutil.copy(template_file, output_file)
        inject_tensor_field(output_file, tensor_data, i)