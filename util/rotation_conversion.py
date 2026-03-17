"""
Rotation conversion utilities for 6D rotation representation.
"""

import functools
from typing import Union, Optional
import numpy as np
import torch
import torch.nn.functional as F


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    
    Args:
        quaternions: quaternions with real part first (w, x, y, z),
            as tensor of shape (..., 4).
    
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions.
    
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    
    Returns:
        quaternions with real part first (w, x, y, z), shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}")
    
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    
    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    
    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    
    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram-Schmidt orthogonalization per Section B of [1].
    
    Args:
        d6: 6D rotation representation, of size (..., 6)
    
    Returns:
        batch of rotation matrices of size (..., 3, 3)
    
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    
    Args:
        matrix: batch of rotation matrices of size (..., 3, 3)
    
    Returns:
        6D rotation representation, of size (..., 6)
    
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis-angle to rotation matrices.
    
    Args:
        axis_angle: Rotations given as a vector in axis-angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis-angle.
    
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    
    Returns:
        Rotations given as a vector in axis-angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis-angle to quaternions.
    
    Args:
        axis_angle: Rotations given as a vector in axis-angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis-angle.
    
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    
    Returns:
        Rotations given as a vector in axis-angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


class RotationTransformer:
    """
    Transformer for converting between different rotation representations.
    Always uses matrix as intermediate representation.
    """
    
    valid_reps = ["axis_angle", "quaternion", "rotation_6d", "matrix"]
    
    def __init__(
        self,
        from_rep: str = "quaternion",
        to_rep: str = "rotation_6d",
    ):
        """
        Initialize rotation transformer.
        
        Args:
            from_rep: Source rotation representation
            to_rep: Target rotation representation
        """
        assert from_rep != to_rep, "Source and target representations must be different"
        assert from_rep in self.valid_reps, f"Invalid from_rep: {from_rep}"
        assert to_rep in self.valid_reps, f"Invalid to_rep: {to_rep}"
        
        self.from_rep = from_rep
        self.to_rep = to_rep
        
        forward_funcs = []
        inverse_funcs = []
        
        # Add conversion to matrix if needed
        if from_rep != "matrix":
            forward_funcs.append(globals()[f"{from_rep}_to_matrix"])
            inverse_funcs.append(globals()[f"matrix_to_{from_rep}"])
        
        # Add conversion from matrix if needed
        if to_rep != "matrix":
            forward_funcs.append(globals()[f"matrix_to_{to_rep}"])
            inverse_funcs.append(globals()[f"{to_rep}_to_matrix"])
        
        # Reverse inverse functions for proper order
        inverse_funcs = inverse_funcs[::-1]
        
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs
    
    @staticmethod
    def _apply_funcs(
        x: Union[np.ndarray, torch.Tensor], 
        funcs: list
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply a sequence of functions to input."""
        is_numpy = isinstance(x, np.ndarray)
        x_tensor = torch.from_numpy(x) if is_numpy else x
        
        for func in funcs:
            x_tensor = func(x_tensor)
        
        return x_tensor.numpy() if is_numpy else x_tensor
    
    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert from source representation to target representation."""
        return self._apply_funcs(x, self.forward_funcs)
    
    def inverse(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert from target representation back to source representation."""
        return self._apply_funcs(x, self.inverse_funcs)
    
    def get_output_dim(self, input_dim: int) -> int:
        """Get output dimension based on input dimension."""
        dim_map = {
            "axis_angle": 3,
            "quaternion": 4,
            "rotation_6d": 6,
            "matrix": 9,  # When flattened
        }
        
        # Verify input dimension matches expected
        expected_from_dim = dim_map[self.from_rep]
        # Account for position (3) and gripper (1 or 2)
        rot_dim = input_dim - 3 - 1  # Assuming 1D gripper
        if rot_dim != expected_from_dim:
            rot_dim = input_dim - 3 - 2  # Try 2D gripper
            if rot_dim != expected_from_dim:
                raise ValueError(
                    f"Input dimension {input_dim} doesn't match expected rotation dim {expected_from_dim} "
                    f"for {self.from_rep} (assuming 3D position and 1-2D gripper)"
                )
        
        # Calculate output dimension
        gripper_dim = input_dim - 3 - expected_from_dim
        output_dim = 3 + dim_map[self.to_rep] + gripper_dim
        return output_dim