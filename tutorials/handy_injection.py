import numpy as np
import os
from foam.foamfield import parse_foam_file

# Define paths
input_file = '../files/handy_injection/examples/TinyS'
output_dir = '../files/handy_injection/temp'
output_file = os.path.join(output_dir, 'TinyS')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate random data of shape (5, 6)
# For a symmetric tensor, the 6 components are [xx, xy, xz, yy, yz, zz]
random_data = np.random.randn(5, 6)

# Parse the existing OpenFOAM file
foam_field = parse_foam_file(input_file)

# Replace only the internal field with the new random data
foam_field.internal_field = random_data.tolist()

# Inject the modified data back to a new file
foam_field.inject(output_file)

print(f"Successfully injected random data into {output_file}")
print("Original data shape:", len(foam_field.internal_field))
print("New data shape:", len(random_data))