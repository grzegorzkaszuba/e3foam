from typing import Union, Dict, List, Optional
import os
import torch
from tensors.base import TensorData
from foam.readers import FoamField, parse_foam_file

def read_foam_field(filepath: str) -> TensorData:
    """
    Read a single OpenFOAM field file and convert to TensorData.
    
    Args:
        filepath: Path to OpenFOAM field file
        
    Returns:
        TensorData with appropriate rank and symmetry
    """
    foam_field = parse_foam_file(filepath)
    
    # Map OpenFOAM types to TensorData properties
    foam_type_map = {
        'volScalarField': {'rank': 0, 'symmetry': 0},
        'volVectorField': {'rank': 1, 'symmetry': 0},
        'volTensorField': {'rank': 2, 'symmetry': 0},
        'volSymmTensorField': {'rank': 2, 'symmetry': 1},
        'volSphericalTensorField': {'rank': 2, 'symmetry': 1},
        'volSkewTensorField': {'rank': 2, 'symmetry': -1}
    }
    
    if foam_field.field_type not in foam_type_map:
        raise ValueError(f"Unsupported field type: {foam_field.field_type}")
        
    props = foam_type_map[foam_field.field_type]
    tensor = foam_field.to_tensor()
    
    # For symmetric tensors, tensor is already in [xx, xy, xz, yy, yz, zz] format
    # No need to reshape to 3x3 and flatten again
    if props['rank'] == 2 and props['symmetry'] == 1:
        return TensorData(tensor, rank=props['rank'], symmetry=props['symmetry'], is_flattened=True)
    
    return TensorData(tensor, rank=props['rank'], symmetry=props['symmetry'])

def read_simulation_case(
    case_dir: str,
    field_paths: List[str],
    time_step: Optional[str] = None
) -> TensorData:
    """
    Read multiple fields from an OpenFOAM case directory.
    
    Args:
        case_dir: Path to OpenFOAM case directory
        field_paths: List of field paths relative to time directory
                    e.g. ['U', 'p', 'R', 'k']
        time_step: Optional specific time step to read from
                  If None, reads from first time step found
                  
    Returns:
        TensorData with all fields combined
    """
    # Find time step directory
    if time_step is None:
        time_dirs = [d for d in os.listdir(case_dir) 
                    if os.path.isdir(os.path.join(case_dir, d)) 
                    and d[0].isdigit()]
        if not time_dirs:
            raise ValueError(f"No time directories found in {case_dir}")
        time_step = sorted(time_dirs)[0]
    
    time_dir = os.path.join(case_dir, time_step)
    
    # Read each field
    tensor_fields = []
    for field_path in field_paths:
        full_path = os.path.join(time_dir, field_path)
        tensor_fields.append(read_foam_field(full_path))
    
    # Combine fields
    return TensorData.cat(tensor_fields)

def tensor_to_foam(tensor_data: TensorData, field_idx: int = 0) -> Dict:
    """
    Convert a TensorData field to foam-compatible format.
    
    Args:
        tensor_data: TensorData object
        field_idx: Which field to convert if multi-field tensor
        
    Returns:
        Dictionary with foam-compatible tensor data
    """
    field = tensor_data.get_field(field_idx) if len(tensor_data.ptr.ptr) > 2 else tensor_data
    
    # Convert based on rank
    rank = field.rank.values[0].item()
    if rank == 0:
        return {"type": "scalar", "data": field.tensor.reshape(-1).tolist()}
    elif rank == 1:
        return {"type": "vector", "data": field.tensor.reshape(-1, 3).tolist()}
    elif rank == 2:
        symmetry = field.symmetry.values[0].item()
        if symmetry == 0:  # No symmetry
            return {"type": "tensor", "data": field.tensor.reshape(-1, 3, 3).tolist()}
        elif symmetry == 1:  # Symmetric
            return {"type": "symmTensor", "data": field.tensor.tolist()}
        else:  # Skew-symmetric
            return {"type": "skewTensor", "data": field.tensor.tolist()}

def foam_to_tensor(
    foam_data: Dict,
    symmetry: int = 0,
    batch_size: int = None
) -> TensorData:
    """
    Convert foam data to TensorData.
    
    Args:
        foam_data: Dictionary with foam tensor data
        symmetry: For rank-2 tensors, specify symmetry type
        batch_size: Optional reshaping to batch size
        
    Returns:
        TensorData object
    """
    data_type = foam_data["type"]
    data = torch.tensor(foam_data["data"])
    
    if batch_size:
        if data_type == "scalar":
            data = data.reshape(batch_size, 1)
        elif data_type == "vector":
            data = data.reshape(batch_size, 3)
        elif data_type in ["tensor", "symmTensor", "skewTensor"]:
            data = data.reshape(batch_size, -1)
    
    # Create TensorData with appropriate properties
    if data_type == "scalar":
        return TensorData(data)
    elif data_type == "vector":
        return TensorData(data)
    elif data_type == "tensor":
        return TensorData(data, symmetry=symmetry)
    elif data_type == "symmTensor":
        return TensorData(data, symmetry=1)
    elif data_type == "skewTensor":
        return TensorData(data, symmetry=-1)

if __name__ == "__main__":
    print("Running OpenFOAM field reader tests...")
    
    # Setup temp directory
    temp_dir = os.path.join(os.path.dirname(__file__), '..', 'files', 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    def print_field_info(foam_field, tensor_data=None, save_path=None):
        """Helper to debug field information"""
        print("\nField Info:")
        print(f"Field type: {foam_field.field_type}")
        print(f"Dimensions: {foam_field.dimensions}")
        print(f"Internal field type: {type(foam_field.internal_field)}")
        print(f"Internal field shape: {len(foam_field.internal_field)} entries")
        print(f"First entry type: {type(foam_field.internal_field[0])}")
        print(f"First entry: {foam_field.internal_field[0]}")
        
        if tensor_data:
            print("\nTensor Info:")
            print(f"Tensor shape: {tensor_data.shape}")
            print(f"Rank: {tensor_data.rank.values[0]}")
            print(f"Symmetry: {tensor_data.symmetry.values[0]}")
            print(f"First value: {tensor_data.tensor[0]}")
        
        # Write field to temp directory
        if save_path:
            foam_field.inject(save_path)
            print(f"\nWrote field to: {save_path}")

    if True:  # Test with synthetic data
        print("\n=== Test 1: Synthetic SymmTensor ===")
        
        # Write synthetic test to temp
        temp_path = os.path.join(temp_dir, "synthetic_symm_tensor")
        
        test_foam_file = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2206                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volSymmTensorField;
    dimensions  [0 2 -2 0 0 0 0];
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   nonuniform List<symmTensor> 
3
(
(1 0.5 0 0.5 2 0)
(2 0 0 2 0 0)
(0.5 0 0 0.5 0 0)
)

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (0 0 0 0 0 0);
    }
}"""
        
        try:
            # First test parsing
            print("\nTesting file parsing...")
            foam_field = parse_foam_file(temp_path)
            print_field_info(foam_field, save_path=temp_path)
            
            # Test tensor conversion
            print("\nTesting tensor conversion...")
            tensor = foam_field.to_tensor()
            print(f"Raw tensor shape: {tensor.shape}")
            print(f"Raw tensor: {tensor}")
            
            # Test full conversion
            print("\nTesting TensorData conversion...")
            tensor_data = read_foam_field(temp_path)
            print_field_info(foam_field, tensor_data, save_path=temp_path)
            
            # Assertions
            assert tensor_data.shape == (3, 6), f"Expected shape (3, 6), got {tensor_data.shape}"
            assert tensor_data.rank.values[0] == 2, "Expected rank 2"
            assert tensor_data.symmetry.values[0] == 1, "Expected symmetry 1"
            
            print("\nSynthetic test passed!")
            
        except Exception as e:
            print(f"\nError in synthetic test: {str(e)}")
            import traceback
            traceback.print_exc()

    if True:  # Test with real OpenFOAM files
        print("\n=== Test 2: Real OpenFOAM Data ===")
        examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
        
        if not os.path.exists(examples_dir):
            print(f"Examples directory not found: {examples_dir}")
        else:
            for filename in os.listdir(examples_dir):
                print(f"\nTesting {filename}:")
                try:
                    example_path = os.path.join(examples_dir, filename)
                    temp_path = os.path.join(temp_dir, filename)
                    
                    foam_field = parse_foam_file(example_path)
                    tensor_data = read_foam_field(example_path)
                    print_field_info(foam_field, tensor_data, save_path=temp_path)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue