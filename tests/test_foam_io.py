import os
import filecmp
import pytest
from e3foam.foam.readers import parse_foam_file

def test_foam_field_roundtrip():
    """Test reading and writing OpenFOAM fields."""
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp')
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Test each example file
    for filename in os.listdir(examples_dir):
        if not filename.endswith('.foam'):  # assuming .foam extension
            continue
            
        example_path = os.path.join(examples_dir, filename)
        temp_path = os.path.join(temp_dir, filename)
        
        # Read and write back
        foam_field = parse_foam_file(example_path)
        foam_field.inject(temp_path)
        
        # Compare files
        assert filecmp.cmp(example_path, temp_path, shallow=False), \
            f"Recreated file {filename} differs from original" 