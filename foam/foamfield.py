"""
OpenFOAM file readers and data structures.
Handles conversion between OpenFOAM and tensor representations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tensors.base import TensorData


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
                if line.isdigit() and array_size is None:
                    a = filepath
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


def compare_foam_fields(ground_truth_path: str, prediction_path: str, relative_tolerance: float = 1e-10) -> Dict[
    str, float]:
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

def plot_case(case_path, variables: List[str], constants: List[Union[TensorData, torch.Tensor]], metrics: List[Callable], 
              show=True, save_path=None):
    """Plot metrics over time for OpenFOAM case.

    Args:
        case_path: Path to OpenFOAM case directory
        variables: List of field names to extract from each time step
        constants: List of reference tensors (either TensorData or torch.Tensor)
        metrics: List of metric functions to compute
        show: Whether to display the plot
        save_path: Optional path to save the plot
    """
    # Convert constants to tensors if they're TensorData
    const_tensors = []
    for const in constants:
        if isinstance(const, TensorData):
            const_tensors.append(const.to_tensor())
        elif isinstance(const, torch.Tensor):
            const_tensors.append(const)
        else:
            raise ValueError(f"Constants must be either TensorData or torch.Tensor, got {type(const)}")
            
    # Step 1: "Assume that each directory whose name is fully an integer or floating-point number 
    # is an appropriate time-step directory"
    def is_numeric_dir(name):
        try:
            float(name)
            return True
        except ValueError:
            return False
            
    time_dirs = sorted([d for d in os.listdir(case_path) 
                       if os.path.isdir(os.path.join(case_path, d)) and is_numeric_dir(d)],
                       key=float)
    
    if not time_dirs:
        raise ValueError(f"No time step directories found in {case_path}")
        
    # Step 2: "For each time-step directory, if it has all the fields listed in the variables, 
    # capture them as FoamFields and extract their main InternalField"
    times = []
    metric_values = [[] for _ in metrics]
    
    for time_dir in time_dirs:
        # Check if all required variables exist
        all_vars_exist = all(os.path.exists(os.path.join(case_path, time_dir, var)) 
                           for var in variables)
        if not all_vars_exist:
            continue
            
        # Load variables as tensors
        var_tensors = []
        for var in variables:
            foam_field = parse_foam_file(os.path.join(case_path, time_dir, var))
            var_tensors.append(foam_field.to_tensor())
            
        # "Now, for each timestep, we have a collection of torch tensors"
        times.append(float(time_dir))
        
        # Compute each metric
        for i, metric_fn in enumerate(metrics):
            # "lambda variables, constants: some_metric"
            value = metric_fn(var_tensors, const_tensors)
            metric_values[i].append(value.item())
            
    # Convert to numpy for plotting
    times = np.array(times)
    metric_values = [np.array(vals) for vals in metric_values]
    
    # "Handle nice formatting for multiple cases, like row by row"
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics), squeeze=False)
    
    for i, (values, ax) in enumerate(zip(metric_values, axes.flat)):
        ax.plot(times, values, 'o-')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Metric {i+1}')
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show(block=True)  # block=True ensures the window stays open
    plt.close()
    
    return times, metric_values


__all__ = ['FoamField', 'parse_foam_file', 'compare_foam_fields', 'plot_case']


if __name__ == "__main__":
    # Add to existing tests
    print("\n=== Test 3: Field Comparison ===")
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'files', 'comparison', 'examples')
    truth_dir = os.path.join(os.path.dirname(__file__), '..', 'files', 'comparison', 'truth')
    
    for filename in os.listdir(examples_dir):
        example_path = os.path.join(examples_dir, filename)
        truth_path = os.path.join(truth_dir, filename)
        
        if os.path.exists(truth_path):
            print(f"\nComparing {filename}:")
            try:
                metrics = compare_foam_fields(example_path, truth_path)
                for name, value in metrics.items():
                    print(f"{name}: {value:.6e}")
            except Exception as e:
                print(f"Error comparing {filename}: {str(e)}")
                continue 