"""
Compare RANS and DNS Reynolds stress tensors to evaluate modeling accuracy.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from foam.readers import compare_foam_fields, parse_foam_file

def analyze_reynolds_stress_error(dns_path: str, rans_path: str):
    """
    Analyze error between RANS and DNS Reynolds stress tensors.
    
    Args:
        dns_path: Path to DNS Reynolds stress tensor (Rdns)
        rans_path: Path to RANS Reynolds stress tensor (Rrans)
    """
    # Calculate error metrics
    metrics = compare_foam_fields(dns_path, rans_path)
    
    print("=== Reynolds Stress Model Evaluation ===")
    print("\nOverall Error Metrics:")
    print("-" * 40)
    for name, value in metrics.items():
        if name.endswith('error'):
            print(f"{name:20s}: {value:10.3e}")
        else:
            print(f"{name:20s}: {value:10.3e}")
    
    # Load tensors for additional analysis
    dns_field = parse_foam_file(dns_path)
    rans_field = parse_foam_file(rans_path)
    
    dns_tensor = dns_field.to_tensor()
    rans_tensor = rans_field.to_tensor()
    
    # Component-wise analysis (for symmetric tensor: xx, xy, xz, yy, yz, zz)
    component_names = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    print("\nComponent-wise RMSE:")
    print("-" * 40)
    for i, comp in enumerate(component_names):
        rmse = torch.sqrt(torch.mean((dns_tensor[:, i] - rans_tensor[:, i])**2))
        print(f"{comp:5s}: {rmse:10.3e}")
    
    # Calculate anisotropy error
    dns_trace = torch.sum(dns_tensor[:, [0,3,5]], dim=1, keepdim=True) / 3  # (xx + yy + zz)/3
    rans_trace = torch.sum(rans_tensor[:, [0,3,5]], dim=1, keepdim=True) / 3
    
    dns_anisotropy = dns_tensor - dns_trace * torch.tensor([1,0,0,1,0,1], dtype=torch.float32)
    rans_anisotropy = rans_tensor - rans_trace * torch.tensor([1,0,0,1,0,1], dtype=torch.float32)
    
    anisotropy_error = torch.mean(torch.abs(dns_anisotropy - rans_anisotropy))
    
    print("\nTurbulence Structure Analysis:")
    print("-" * 40)
    print(f"Mean Anisotropy Error: {anisotropy_error:10.3e}")
    print(f"Mean TKE Error: {torch.mean(torch.abs(dns_trace - rans_trace)):10.3e}")

if __name__ == "__main__":
    # Setup paths
    examples_dir = Path(__file__).parent.parent / "examples"
    data_dir = Path(__file__).parent.parent.parent / "data" / "macedo" / "periodic-hills" / "0p5" / "0"
    
    # Test with example files
    print("\nTesting with example files:")
    dns_path = examples_dir / "Rdns"
    rans_path = examples_dir / "Rrans"
    
    if dns_path.exists() and rans_path.exists():
        analyze_reynolds_stress_error(str(dns_path), str(rans_path))
    
    # Test with full dataset if available
    print("\nTesting with full dataset:")
    dns_path = data_dir / "Rdns"
    rans_path = data_dir / "Rrans"
    
    if dns_path.exists() and rans_path.exists():
        analyze_reynolds_stress_error(str(dns_path), str(rans_path))
    else:
        print("Full dataset not found. Skipping...") 