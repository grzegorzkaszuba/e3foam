"""
OpenFOAM file readers and data structures.
Handles conversion between OpenFOAM and tensor representations.
"""

# We'll migrate and improve OpenFOAM handling here

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add parent dir to path
from tensors.base import TensorData


@dataclass
class FoamField:
    # File structure elements
    header: str  # The ASCII art header
    separator: str  # The // * * * line
    footer: str  # The // ***** line

    # FoamFile section
    version: str
    format_type: str
    field_type: str  # class in OpenFOAM terminology
    location: str
    object_name: str

    # Field data
    dimensions: str  # Keep raw string to preserve exact format [ 0 2 -2 0 0 0 0 ]
    internal_field_type: str  # e.g. "nonuniform List<symmTensor>"
    internal_field_size: int
    internal_field_data: List[str]  # Keep raw strings to preserve exact formatting

    # Newlines preservation
    dimensions_newlines: str  # Store newlines before dimensions
    internal_field_newlines: str  # Store newlines before internal field

    # Optional fields with defaults
    boundary_field: Optional[Dict[str, Dict[str, Union[str, Dict]]]] = None
    foam_spacing: Dict[str, str] = None  # Spacing in FoamFile section
    separate_closing: bool = False  # Whether ) and ; are on separate lines

    def inject(self, filepath: str):
        """Write field to file, preserving exact format"""
        with open(filepath, 'w') as f:
            # Write header and separator - they already have newlines
            f.write(self.header)
            f.write(self.separator)

            # Write FoamFile section with original spacing
            f.write('FoamFile\n{\n')
            spacing = self.foam_spacing or {'version': '     ', 'format': '      ', 'class': '       ',
                                            'location': '    ', 'object': '      '}

            f.write(f'    version{spacing.get("version", "     ")}{self.version};\n')
            f.write(f'    format{spacing.get("format", "      ")}{self.format_type};\n')
            f.write(f'    class{spacing.get("class", "       ")}{self.field_type};\n')
            f.write(f'    location{spacing.get("location", "    ")}"{self.location}";\n')
            f.write(f'    object{spacing.get("object", "      ")}{self.object_name};\n')
            f.write('}\n')  # Close FoamFile section
            f.write(self.dimensions_newlines)  # Write stored newlines
            f.write(f'dimensions{self.dimensions}')

            f.write(self.internal_field_newlines)  # Write stored newlines
            f.write(f'internalField{self.internal_field_type}')
            f.write(f'{self.internal_field_size}\n')
            f.write('(\n')
            for line in self.internal_field_data:
                f.write(line)  # line already has newline

            # Write closing parenthesis and semicolon
            if self.separate_closing:
                f.write(')\n;\n')
            else:
                f.write(');\n')

            # Write boundary field if present
            if self.boundary_field:
                f.write('\nboundaryField\n{\n')
                # TODO: Implement boundary field writing preserving exact format
                f.write('}\n')

            f.write(self.footer)

    def get_internal_field_array(self) -> torch.Tensor:
        """Extract internal field data as a tensor array.

        Returns:
            torch.Tensor: Array of shape (N, M) where:
                N is the number of data points (internal_field_size)
                M is the number of components per point (e.g. 6 for symmTensor)
        """
        data = []
        for line in self.internal_field_data:
            # Remove parentheses and split by spaces
            values = line.strip('()\n').split()
            # Convert to floats
            data.append([float(x) for x in values])

        return torch.tensor(data, dtype=torch.float)


def parse_foam_file(filepath: str) -> FoamField:
    """Parse OpenFOAM field file preserving exact format"""
    with open(filepath, 'r') as f:
        content = f.readlines()

    # Find key sections - header start and end
    header_end = next(i for i, line in enumerate(content) if
                      '*---------------------------------------------------------------------------*/' in line)
    separator_line = next(i for i, line in enumerate(content) if
                          '* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *' in line)

    # Extract header and separator - preserve exact format
    header_lines = content[:header_end + 1]  # Get all header lines including the closing line
    header = ''.join(header_lines)  # Keep them exactly as they are - no extra newline at start
    separator = content[separator_line]

    # Find FoamFile section and parse with original spacing
    foamfile_start = next(i for i, line in enumerate(content) if 'FoamFile' in line)
    foamfile_end = next(i for i, line in enumerate(content[foamfile_start:]) if '}' in line) + foamfile_start

    # Parse FoamFile section with original spacing
    foamfile_lines = content[foamfile_start + 2:foamfile_end]  # Skip 'FoamFile' and '{'
    foam_dict = {}
    foam_spacing = {}  # Store original spacing
    for line in foamfile_lines:
        if ';' in line:
            line_without_semicolon = line.split(';')[0]
            if not line_without_semicolon.strip():
                continue

            # Split by first space sequence
            parts = line_without_semicolon.split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                foam_dict[key] = value.strip().strip('"')
                # Calculate spacing - preserve exact spacing
                spacing = line.split(key)[1].split(value.strip())[0]
                foam_spacing[key] = spacing

    # Find dimensions and count newlines before it
    dim_line = next(i for i, line in enumerate(content) if 'dimensions' in line)
    dimensions_newlines = ''.join(content[foamfile_end + 1:dim_line])  # Get all lines between FoamFile and dimensions
    prefix, rest = content[dim_line].split('dimensions', 1)
    dimensions = rest  # Keep all whitespace

    # Find internal field and count newlines before it
    internal_start = next(i for i, line in enumerate(content) if 'internalField' in line)
    internal_field_newlines = ''.join(
        content[dim_line + 1:internal_start])  # Get all lines between dimensions and internal field
    internal_line = content[internal_start]
    prefix, rest = internal_line.split('internalField', 1)
    internal_type = rest  # Keep all whitespace
    size = int(content[internal_start + 1].strip())

    # Get data lines - preserve all whitespace
    data_start = next(i for i, line in enumerate(content[internal_start:]) if '(' in line) + internal_start

    # Find the end by looking for either ');' or ') ;' pattern
    data_end = None
    separate_closing = False  # Track if ) and ; are on separate lines

    for i, line in enumerate(content[data_start:]):
        if line.strip() == ')':  # Just closing parenthesis
            for j, next_line in enumerate(content[data_start + i + 1:]):
                if ';' in next_line:  # Found semicolon on a later line
                    data_end = i + data_start
                    separate_closing = True
                    break
            if separate_closing:
                break
        elif ';' in line:  # Found a semicolon
            if ')' in line:  # Case 1: ');' on same line
                data_end = i + data_start
                break
            elif any(')' in prev_line for prev_line in
                     content[data_start:i + data_start]):  # Case 2: ')' and ';' separate
                data_end = i + data_start
                separate_closing = True
                break

    if data_end is None:
        raise ValueError("Could not find end of internal field data")

    # Get all data lines between ( and ), excluding the parentheses lines
    internal_data = []
    for line in content[data_start + 1:data_end]:
        if line.strip() and line.strip() != ')':  # Only use strip() for checking emptiness
            internal_data.append(line)  # Keep original whitespace

    # Everything after the semicolon until boundaryField is footer
    footer_start = next(i for i, line in enumerate(content[data_end:]) if ';' in line) + data_end

    # Check for boundary field section
    boundary_field = None
    boundary_start = None

    for i, line in enumerate(content[footer_start + 1:]):
        if 'boundaryField' in line:
            boundary_start = i + footer_start + 1
            break

    if boundary_start is not None:
        # Parse boundary field section
        boundary_field = {}
        current_boundary = None
        current_dict = {}
        in_array = False
        array_data = []

        for line in content[boundary_start + 2:]:  # Skip 'boundaryField' and '{'
            stripped = line.strip()

            if in_array:
                if stripped == ')':
                    in_array = False
                    current_dict['value_data'] = array_data
                    array_data = []
                elif stripped and stripped != '(':
                    array_data.append(line)  # Keep original line with whitespace
                continue

            if not stripped:
                continue

            if stripped == '}':  # End of boundaryField
                if current_boundary:
                    boundary_field[current_boundary] = current_dict
                break

            if stripped.endswith('{'):  # Start of new boundary
                if current_boundary:
                    boundary_field[current_boundary] = current_dict
                current_boundary = stripped[:-1].strip()
                current_dict = {}
            elif 'List<' in line and '(' in line:  # Start of array data
                in_array = True
                key, value = line.split(maxsplit=1)
                current_dict[key] = value.split('(')[0].strip()
            elif ';' in line:  # Key-value pair
                key, value = line.split(maxsplit=1)
                current_dict[key] = value.rstrip(';').strip()

        # Footer is everything after the last }
        footer = ''.join(content[next(i for i, line in enumerate(content) if line.strip() == '}'):])
    else:
        # No boundary field - footer is everything after the semicolon
        footer = ''.join(content[footer_start + 1:])

    return FoamField(
        header=header,
        separator=separator,
        footer=footer,
        version=foam_dict['version'],
        format_type=foam_dict['format'],
        field_type=foam_dict['class'],
        location=foam_dict['location'],
        object_name=foam_dict['object'],
        dimensions=dimensions,
        internal_field_type=internal_type,
        internal_field_size=size,
        internal_field_data=internal_data,
        boundary_field=boundary_field,  # Will be None for both files
        foam_spacing=foam_spacing,
        separate_closing=separate_closing,
        dimensions_newlines=dimensions_newlines,
        internal_field_newlines=internal_field_newlines,
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
    if len(gt_field.internal_field_data) != len(pred_field.internal_field_data):
        raise ValueError(
            f"Field sizes don't match: {len(gt_field.internal_field_data)} vs {len(pred_field.internal_field_data)}")

    # Convert to tensors
    gt_tensor = torch.tensor([float(x) for x in gt_field.internal_field_data], dtype=torch.float)
    pred_tensor = torch.tensor([float(x) for x in pred_field.internal_field_data], dtype=torch.float)

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


# todo
# import matplotlib.pyplot as plt
from typing import Callable


def plot_case(case_path, variables: List['str'], constants: List[TensorData], metrics: List[Callable], show=True,
              save_path=None):
    '''
    Args:
        case_path: path to the foam case: a directory in which time steps are stored
        fields: names of fields that shall be captured from each time step
        constants: TensorData objects that store information from fields that are not parsed each time, e.g. ground truth values
        metrics: values to be computed from the captured fields, defined as callables: conveniently, lambda functions
    Returns: None

    write a function that scans the foam case. Assume that each directory whose name is fully an integer or floating-point
    number is an appropriate time-step directory. For each time-step directory, if it has all the fields listed in the
    "variables", capture them as FoamFields and extract their main InternalField - do not bother with alternative
    array fields like BoundaryFields.
    Define a way to compute metrics based on those fields. In general, they will be aggregated, e.g. torch loss metrics,
    because the intended purpose of this function is to compute errors
    Constants are alternative fields, captured only once. For example: if you have 20 different timesteps and want to
    visualize flow velocity error, you need ground truth. They use TensorData objects or FoamField objects: each FoamField's
    internal field can be cast to TensorData anyway, which we intend to do. Handle the preprocessing of each single one to
    go from FoamField, through Tensordata, to finally extract a torch tensor. Do the same for the variables.

    Mind the format for plotting. Our approach is build upon torch mostly, but matplotlib needs a cast to numpy.

    Now, given the data is captured, you can compute metrics to plot as given. We no longer have a list of strings to work with. Now, for each timestep, we have a collection of torch tensors.

    I would like to propose a constant format for those, which goes like this:
    lambda variables, constants: some_metric

    For example, if you've captured variables S, R, U, k and the ground truth values U, tau, you can assign a plot of flow field error modeling by providing the following callable:
    lambda vars, consts: torch.nn.functional.mse_loss(vars[3], consts[0])

    finally, if show==True, we show the plot. If save path, we save the plot in the save path

    Handle nice formatting for multiple cases, like row by row,
    '''
    pass


def check_file_identity(file1: str, file2: str) -> bool:
    """
    Check if two files are identical line by line.

    Args:
        file1: Path to first file
        file2: Path to second file

    Returns:
        True if files are identical, False otherwise
    """
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    if len(lines1) != len(lines2):
        print(f"Files have different number of lines: {len(lines1)} vs {len(lines2)}")
        return False

    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1 != line2:
            print(f"Line {i + 1} differs:")
            print(f"  File 1: {repr(line1)}")
            print(f"  File 2: {repr(line2)}")
            return False

    return True
