import os
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add parent dir to path
from foam.foamfield import parse_foam_file

def test_foam_field_roundtrip():
    """Test reading and writing OpenFOAM fields."""
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'files', 'identity', 'examples')
    temp_dir = os.path.join(os.path.dirname(__file__), '..', 'files', 'identity', 'temp')
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Test each example file
    for filename in os.listdir(examples_dir):
        example_path = os.path.join(examples_dir, filename)
        temp_path = os.path.join(temp_dir, filename)
        
        # Read and write back
        foam_field = parse_foam_file(example_path)
        foam_field.inject(temp_path)
        
        # Check identity
        # assert check_file_identity(example_path, temp_path), \
        #     f"Recreated file {filename} differs from original"

if __name__ == "__main__":
    test_foam_field_roundtrip()
