import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add parent dir to path
from foam.foamfield import parse_foam_file

def test_print_array():
    # Get path to TinyS_NB example file
    file_path = os.path.join(os.path.dirname(__file__), '..', 'files', 'identity', 'examples', 'TinyS_NB')
    
    # Parse the file
    foam_field = parse_foam_file(file_path)
    
    # Convert to tensor and print
    tensor = foam_field.to_tensor()
    print("\nField type:", foam_field.field_type)
    print("Dimensions:", foam_field.dimensions)
    print("Internal field shape:", tensor.shape)
    #print("Internal field values:\n", tensor)

if __name__ == "__main__":
    test_print_array() 