import torch

def project_tensor_to_2d(tensor, plane='xy', preserve_trace=True):
    """
    Project a 3D tensor to 2D while optionally preserving the trace.
    
    Args:
        tensor: Tensor in Cartesian coordinates [batch_size, ...]
        plane: Projection plane ('xy', 'xz', or 'yz')
        preserve_trace: Whether to preserve the 3D trace
        
    Returns:
        Projected tensor in Cartesian coordinates
    """
    # Determine which axis to zero out
    if plane == 'xy':
        zero_axis = 2  # z-axis
    elif plane == 'xz':
        zero_axis = 1  # y-axis
    elif plane == 'yz':
        zero_axis = 0  # x-axis
    else:
        raise ValueError(f"Unknown plane: {plane}. Use 'xy', 'xz', or 'yz'.")
    
    # Get the remaining axes
    axes = [0, 1, 2]
    axes.remove(zero_axis)
    axis1, axis2 = axes
    
    # Handle different tensor ranks
    if tensor.dim() == 2 and tensor.shape[1] in [1, 3, 6, 9]:
        # Flattened tensor format
        if tensor.shape[1] == 1:  # Scalar
            return tensor  # No projection needed
            
        elif tensor.shape[1] == 3:  # Vector [x, y, z]
            result = tensor.clone()
            result[:, zero_axis] = 0
            return result
            
        elif tensor.shape[1] == 6:  # Symmetric tensor [xx, xy, xz, yy, yz, zz]
            result = tensor.clone()
            
            # Map indices to positions in the flattened tensor
            idx_map = {
                (0, 0): 0,  # xx
                (0, 1): 1,  # xy
                (0, 2): 2,  # xz
                (1, 1): 3,  # yy
                (1, 2): 4,  # yz
                (2, 2): 5   # zz
            }
            
            # Zero out components involving the zero_axis
            for i in range(3):
                for j in range(i, 3):
                    if i == zero_axis or j == zero_axis:
                        flat_idx = idx_map.get((min(i, j), max(i, j)))
                        if flat_idx is not None:
                            if preserve_trace and i == j == zero_axis:
                                # Distribute the diagonal component to preserve trace
                                diag_value = tensor[:, flat_idx] / 2
                                result[:, idx_map[(axis1, axis1)]] += diag_value
                                result[:, idx_map[(axis2, axis2)]] += diag_value
                            result[:, flat_idx] = 0
            
            return result
            
        elif tensor.shape[1] == 9:  # General tensor [xx, xy, xz, yx, yy, yz, zx, zy, zz]
            result = tensor.clone()
            
            # Reshape to 3x3
            batch_size = tensor.shape[0]
            reshaped = result.view(batch_size, 3, 3)
            
            # Handle trace preservation for diagonal component
            if preserve_trace:
                diag_value = reshaped[:, zero_axis, zero_axis] / 2
                reshaped[:, axis1, axis1] += diag_value
                reshaped[:, axis2, axis2] += diag_value
            
            # Zero out the row and column
            reshaped[:, zero_axis, :] = 0
            reshaped[:, :, zero_axis] = 0
            
            # Flatten back
            return reshaped.view(batch_size, 9)
    
    elif tensor.dim() >= 3 and tensor.shape[-2:] == (3, 3):
        # Tensor in matrix form [..., 3, 3]
        result = tensor.clone()
        
        # Handle trace preservation for diagonal component
        if preserve_trace:
            diag_value = result[..., zero_axis, zero_axis] / 2
            result[..., axis1, axis1] += diag_value
            result[..., axis2, axis2] += diag_value
        
        # Zero out the row and column
        result[..., zero_axis, :] = 0
        result[..., :, zero_axis] = 0
        
        return result
    
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
