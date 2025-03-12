"""
OpenFOAM file readers and data structures.
Handles conversion between OpenFOAM and tensor representations.
"""

# We'll migrate and improve OpenFOAM handling here 

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
import os

@dataclass
class FoamField:
    """
    Represents an OpenFOAM field file structure.
    
    Attributes:
        field_type: OpenFOAM field class (e.g., volVectorField)
        dimensions: Physical dimensions [kg m s K mol A cd]
        internal_field: Main field data
        boundary_field: Optional boundary conditions
        format_version: OpenFOAM format version
        format_type: ascii/binary
        location: Original file path
    """
    field_type: str
    dimensions: List[int]
    internal_field: Union[List[float], List[List[float]]]
    boundary_field: Optional[Dict[str, Dict]] = None
    format_version: str = "2.0"
    format_type: str = "ascii"
    location: Optional[str] = None
    
    @property
    def size(self) -> int:
        """Number of entries in internal field."""
        return len(self.internal_field)
    
    @property
    def is_uniform(self) -> bool:
        """Whether the field has uniform values."""
        return len(self.internal_field) == 1
    
    def to_tensor(self) -> torch.Tensor:
        """Convert internal field to tensor."""
        print(f"DEBUG: to_tensor called")
        print(f"DEBUG: internal_field type: {type(self.internal_field)}")
        print(f"DEBUG: first value type: {type(self.internal_field[0])}")
        print(f"DEBUG: first few values: {self.internal_field[:3]}")
        
        # Check if first entry is a list/tuple (vector/tensor) or scalar
        if isinstance(self.internal_field[0], (list, tuple)):
            # Vector/Tensor case - already in list of lists format
            return torch.tensor(self.internal_field, dtype=torch.float)
        else:
            # Scalar case - reshape to column
            return torch.tensor(self.internal_field, dtype=torch.float).reshape(-1, 1)

    def format_float(self, value: float) -> str:
        """Format float values according to OpenFOAM conventions."""
        sci_notation_threshold = 1e-04
        
        if abs(value) < sci_notation_threshold:
            if abs(value) < 1e-15:
                return "0"
            formatted = f"{value:.5e}"
            return formatted.replace("e+", "e")
        return f"{value:.9f}"

    def inject(self, output_path: Optional[str] = None) -> None:
        """Write field data back to OpenFOAM file."""
        if output_path is None:
            if self.location is None:
                raise ValueError("No file path specified for injection")
            output_path = self.location
            
        # Format dimensions
        dims_str = f"[{' '.join(str(d) for d in self.dimensions)}]"
        
        # Format internal field values
        if isinstance(self.internal_field[0], (list, tuple)):
            # Vector/Tensor
            values_str = "\n".join(
                f"({' '.join(self.format_float(x) for x in row)})"
                for row in self.internal_field
            )
        else:
            # Scalar
            values_str = "\n".join(
                self.format_float(x) for x in self.internal_field
            )
            
        # Format boundary field
        boundary_str = ""
        if self.boundary_field:
            boundary_str = "boundaryField\n{\n"
            for name, props in self.boundary_field.items():
                boundary_str += f"    {name}\n    {{\n"
                for key, value in props.items():
                    boundary_str += f"        {key}    {value};\n"
                boundary_str += "    }\n"
            boundary_str += "}"
            
        # Write file
        with open(output_path, 'w') as f:
            f.write(f"""FoamFile
{{
    version     {self.format_version};
    format      {self.format_type};
    class       {self.field_type};
    location    "0";
    object      {os.path.basename(output_path)};
}}

dimensions      {dims_str};

internalField   nonuniform List<{self.field_type.replace('vol', '')}> 
{len(self.internal_field)}
(
{values_str}
)

{boundary_str}
""")

def parse_foam_file(filepath: str) -> FoamField:
    """Parse OpenFOAM field file into FoamField object."""
    field_type = None
    dimensions = None
    values = []
    boundary_data = {}
    in_internal_field = False
    in_boundary_field = False
    current_boundary = None
    is_vector_or_tensor = None
    array_size = None  # Track array size separately
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Get field type
            if line.startswith('class'):
                field_type = line.split()[-1].rstrip(';')
                is_vector_or_tensor = 'vector' in field_type.lower() or 'tensor' in field_type.lower()
                continue
                
            # Get dimensions
            if line.startswith('dimensions'):
                dims = line.split('[')[1].split(']')[0]
                dimensions = [int(d) for d in dims.split()]
                continue
                
            # Parse internal field values
            if line.startswith('internalField'):
                in_internal_field = True
                continue
                
            if in_internal_field:
                if line == ')':
                    in_internal_field = False
                    continue
                if line == '(':
                    continue
                # Get array size but don't parse it as value
                if line.isdigit():
                    array_size = int(line)
                    continue
                    
                try:
                    if is_vector_or_tensor:
                        if line.startswith('('):
                            value = [float(x) for x in line.strip('()').split()]
                            values.append(value)
                    else:
                        # For scalar fields, skip lines with parentheses
                        if not line.startswith('('):
                            value = float(line)
                            values.append(value)
                except ValueError:
                    continue
            
            # Parse boundary field
            if line.startswith('boundaryField'):
                in_boundary_field = True
                in_internal_field = False
                continue
                
            if in_boundary_field:
                if line == '{':
                    continue
                if line == '}':
                    if current_boundary:
                        in_boundary_field = False
                        current_boundary = None
                    continue
                    
                # New boundary section
                if not line.startswith(' ') and line.endswith('{'):
                    current_boundary = line.rstrip('{').strip()
                    boundary_data[current_boundary] = {}
                    continue
                
                # Boundary properties
                if current_boundary and '    ' in line:
                    key, value = [x.strip() for x in line.split(maxsplit=1)]
                    value = value.rstrip(';')
                    boundary_data[current_boundary][key] = value
    
    # Verify array size matches
    if array_size is not None and len(values) != array_size:
        raise ValueError(f"Expected {array_size} values, got {len(values)}")

    return FoamField(
        field_type=field_type,
        dimensions=dimensions,
        internal_field=values,
        boundary_field=boundary_data if boundary_data else None,
        location=filepath
    ) 

def compare_foam_fields(ground_truth_path: str, prediction_path: str, relative_tolerance: float = 1e-10) -> Dict[str, float]:
    """
    Compare two OpenFOAM fields and calculate error metrics.
    
    Args:
        ground_truth_path: Path to ground truth OpenFOAM field
        prediction_path: Path to predicted OpenFOAM field
        relative_tolerance: Tolerance for MAPE calculation to avoid division by zero
        
    Returns:
        Dictionary with error metrics:
            - MSE: Mean Squared Error
            - RMSE: Root Mean Squared Error
            - MAE: Mean Absolute Error
            - MAPE: Mean Absolute Percentage Error
            - max_error: Maximum absolute error
            - max_relative_error: Maximum relative error
    """
    # Read both fields
    gt_field = parse_foam_file(ground_truth_path)
    pred_field = parse_foam_file(prediction_path)
    
    # Verify fields are compatible
    if gt_field.field_type != pred_field.field_type:
        raise ValueError(f"Field types don't match: {gt_field.field_type} vs {pred_field.field_type}")
    if len(gt_field.internal_field) != len(pred_field.internal_field):
        raise ValueError(f"Field sizes don't match: {len(gt_field.internal_field)} vs {len(pred_field.internal_field)}")
    
    # Convert to tensors
    gt_tensor = gt_field.to_tensor()
    pred_tensor = pred_field.to_tensor()
    
    # Calculate errors
    abs_error = torch.abs(gt_tensor - pred_tensor)
    squared_error = abs_error ** 2
    
    # Handle relative errors carefully to avoid division by zero
    # Add small tolerance relative to the data scale
    scale = torch.mean(torch.abs(gt_tensor))
    denom = torch.maximum(torch.abs(gt_tensor), torch.full_like(gt_tensor, relative_tolerance * scale))
    relative_error = abs_error / denom
    
    # Calculate metrics
    metrics = {
        "MSE": torch.mean(squared_error).item(),
        "RMSE": torch.sqrt(torch.mean(squared_error)).item(),
        "MAE": torch.mean(abs_error).item(),
        "MAPE": torch.mean(relative_error * 100).item(),  # as percentage
        "max_error": torch.max(abs_error).item(),
        "max_relative_error": torch.max(relative_error * 100).item(),  # as percentage
    }
    
    return metrics

if __name__ == "__main__":
    # Add to existing tests
    print("\n=== Test 3: Field Comparison ===")
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp')
    
    for filename in os.listdir(examples_dir):
        example_path = os.path.join(examples_dir, filename)
        temp_path = os.path.join(temp_dir, filename)
        
        if os.path.exists(temp_path):
            print(f"\nComparing {filename}:")
            try:
                metrics = compare_foam_fields(example_path, temp_path)
                for name, value in metrics.items():
                    print(f"{name}: {value:.6e}")
            except Exception as e:
                print(f"Error comparing {filename}: {str(e)}")
                continue 