import os
import sys
import torch
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import after path is set
from foam.foamfield import plot_case, parse_foam_file
from tensors.base import TensorData

def test_plot_case():
    # Setup paths
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'files', 'plotting')
    truth_path = os.path.join(base_dir, 'truth', 'SimpleS')
    examples_path = os.path.join(base_dir, 'examples')
    
    # Load ground truth as constant - allow both TensorData and torch.Tensor
    try:
        truth_tensor = TensorData.from_foam(truth_path).to_tensor()
    except:
        # If TensorData fails, try loading directly as FoamField and convert to tensor
        truth_tensor = parse_foam_file(truth_path).to_tensor()
        
    constants = [truth_tensor]
    
    # Define metrics to plot
    metrics = [
        lambda vars, consts: torch.nn.functional.mse_loss(vars[0], consts[0]),  # MSE loss
        lambda vars, consts: torch.abs(vars[0] - consts[0]).mean(),  # MAE loss
    ]
    
    # Test plotting
    try:
        plot_case(
            case_path=examples_path,
            variables=['SimpleS'],  # Look for SimpleS field
            constants=constants,
            metrics=metrics,
            show=False,
            save_path='test_plot.png'
        )
        print("Successfully plotted case")
    except Exception as e:
        print(f"Error plotting case: {str(e)}")

    # Test error case (8.21 has wrong field name)
    try:
        plot_case(
            case_path=os.path.join(examples_path, '8.21'),
            variables=['SimpleS'],
            constants=constants,
            metrics=metrics,
            show=False
        )
        print("Error: Should have failed on wrong field name")
    except Exception as e:
        print(f"Successfully caught error: {str(e)}")

if __name__ == "__main__":
    import time
    test_plot_case()
    time.sleep(20)