"""Code derived from https://github.com/StanfordVL/perls2 and https://github.com/ARISE-Initiative/robomimic and https://github.com/ARISE-Initiative/robosuite

Utility functions for controlling the robot.
"""

import math

import torch
from torch.nn import functional as F
from ipdb import set_trace as bp


@torch.jit.script
def opspace_matrices(mass_matrix, J_full):
    """Compute the lambda and nullspace matrices for the operational space control."""
    # Optimize code above.
    lambda_full_inv = torch.matmul(J_full, torch.linalg.solve(mass_matrix, J_full.T))

    # take the inverses, but zero out small singular values for stability
    svd_u, svd_s, svd_v = torch.linalg.svd(lambda_full_inv)
    singularity_threshold = 0.05
    svd_s_inv = torch.tensor(
        [0.0 if x < singularity_threshold else float(1.0 / x) for x in svd_s]
    ).to(mass_matrix.device)
    lambda_full = svd_v.T.matmul(torch.diag(svd_s_inv)).matmul(svd_u.T)

    # nullspace
    Jbar = torch.linalg.solve(mass_matrix, J_full.t()).matmul(lambda_full)
    nullspace_matrix = torch.eye(J_full.shape[-1], J_full.shape[-1]).to(
        mass_matrix.device
    ) - torch.matmul(Jbar, J_full)

    return lambda_full, nullspace_matrix


@torch.jit.script
def sign(x: float, epsilon: float = 0.01):
    """Get the sign of a number"""
    if x > epsilon:
        return 1.0
    elif x < -epsilon:
        return -1.0
    return 0.0


def quaternion_multiply(a, b):
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part last.
        b: Quaternions as tensor of shape (..., 4), real part last.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part last.
        b: Quaternions as tensor of shape (..., 4), real part last.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4), real part last.
    """
    ax, ay, az, aw = torch.unbind(a, -1)
    bx, by, bz, bw = torch.unbind(b, -1)
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    ow = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack((ox, oy, oz, ow), -1)


def proprioceptive_quat_to_6d_rotation(robot_state: torch.Tensor) -> torch.Tensor:
    """
    Convert the 14D proprioceptive state space to 16D state space.

    Parts:
        - 3D position
        - 4D quaternion rotation
        - 3D linear velocity
        - 3D angular velocity
        - 1D gripper width

    Rotation 4D quaternion -> 6D vector represention

    Accepts any number of leading dimensions.
    """
    # assert robot_state.shape[-1] == 14, "Robot state must be 14D"

    # Get each part of the robot state
    pos = robot_state[..., :3]  # (x, y, z)
    ori_quat = robot_state[..., 3:7]  # (x, y, z, w)
    pos_vel = robot_state[..., 7:10]  # (x, y, z)
    ori_vel = robot_state[..., 10:13]  # (x, y, z)
    gripper = robot_state[..., 13:]  # (width)

    # Convert quaternion to 6D rotation
    ori_6d = isaac_quat_to_rot_6d(ori_quat)

    # Concatenate all parts
    robot_state_6d = torch.cat([pos, ori_6d, pos_vel, ori_vel, gripper], dim=-1)

    return robot_state_6d


def isaac_quat_to_rot_6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Converts IsaacGym quaternion to rotation 6D."""
    # Move the real part from the back to the front
    # quat_wxyz = isaac_quat_to_pytorch3d_quat(quat_xyzw)

    # Convert each quaternion to a rotation matrix
    rot_mats = quaternion_to_matrix(quat_xyzw)

    # Extract the first two columns of each rotation matrix
    rot_6d = matrix_to_rotation_6d(rot_mats)

    return rot_6d


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def rotation_6d_to_quaternion_xyzw(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to quaternions.
    Args:
        rot_6d: 6D rotation representation, of size (*, 6)

    Returns:
        batch of quaternions of size (*, 4) with real part last
    """
    rot_mats = rotation_6d_to_matrix(rot_6d)
    return matrix_to_quaternion_xyzw(rot_mats)


def quaternion_to_rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts quaternions to 6D rotation representation by Zhou et al. [1]
    by converting the quaternion to a rotation matrix and then to 6D.
    Args:
        quat: batch of quaternions of size (*, 4)

    Returns:
        6D rotation representation, of size (*, 6)
    """
    rot_mats = quaternion_to_matrix(quat)
    return matrix_to_rotation_6d(rot_mats)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            last, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([-1, -1, -1, 1])


def standardize_quaternion(quaternions):
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., -1:] < 0, -quaternions, quaternions)


def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
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
        [axis_angle * sin_half_angles_over_angles, torch.cos(half_angles)], dim=-1
    )
    return quaternions


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion_xyzw(matrix))


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., :-1], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., -1:])
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
    return quaternions[..., :-1] / sin_half_angles_over_angles


@torch.jit.script
def nullspace_torques(
    mass_matrix: torch.Tensor,
    nullspace_matrix: torch.Tensor,
    initial_joint: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_kp: float = 10,
):
    """
    For a robot with redundant DOF(s), a nullspace exists which is orthogonal to the remainder of the controllable
    subspace of the robot's joints. Therefore, an additional secondary objective that does not impact the original
    controller objective may attempt to be maintained using these nullspace torques.
    This utility function specifically calculates nullspace torques that attempt to maintain a given robot joint
    positions @initial_joint with zero velocity using proportinal gain @joint_kp
    :Note: @mass_matrix, @nullspace_matrix, @joint_pos, and @joint_vel should reflect the robot's state at the current
    timestep
    Args:
        mass_matrix (torch.tensor): 2d array representing the mass matrix of the robot
        nullspace_matrix (torch.tensor): 2d array representing the nullspace matrix of the robot
        initial_joint (torch.tensor): Joint configuration to be used for calculating nullspace torques
        joint_pos (torch.tensor): Current joint positions
        joint_vel (torch.tensor): Current joint velocities
        joint_kp (float): Proportional control gain when calculating nullspace torques
    Returns:
          torch.tensor: nullspace torques
    """
    # kv calculated below corresponds to critical damping
    joint_kv = torch.sqrt(joint_kp) * 2
    # calculate desired torques based on gains and error
    pose_torques = torch.matmul(
        mass_matrix, (joint_kp * (initial_joint - joint_pos) - joint_kv * joint_vel)
    )
    # map desired torques to null subspace within joint torque actuator space
    nullspace_torques = torch.matmul(nullspace_matrix.t(), pose_torques)
    return nullspace_torques


@torch.jit.script
def cross_product(vec1, vec2):
    """Efficient cross product function"""
    mat = torch.tensor(
        (
            [0.0, float(-vec1[2]), float(vec1[1])],
            [float(vec1[2]), 0.0, float(-vec1[0])],
            [float(-vec1[1]), float(vec1[0]), 0.0],
        )
    ).to(vec1.device)
    return torch.matmul(mat, vec2)


@torch.jit.script
def orientation_error(desired, current):
    """Optimized function to determine orientation error from matrices"""

    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (
        cross_product(rc1, rd1) + cross_product(rc2, rd2) + cross_product(rc3, rd3)
    )
    return error


@torch.jit.script
def quat_conjugate(a):
    """
    Compute the conjugate of a quaternion
    Quaternions are represented as (x, y, z, w)
    """
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_mul(a, b):
    """
    Multiply two quaternions
    Quaternions are represented as (x, y, z, w)
    """
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_xyzw_to_wxyz(quat_xyzw):
    """
    Convert quaternions from (x, y, z, w) order to (w, x, y, z) order.

    Args:
        quat_xyzw (torch.Tensor): Quaternions in (x, y, z, w) order, shape (..., 4).

    Returns:
        torch.Tensor: Quaternions in (w, x, y, z) order, shape (..., 4).
    """
    inds = torch.tensor([3, 0, 1, 2], dtype=torch.long, device=quat_xyzw.device)
    return torch.index_select(quat_xyzw, dim=-1, index=inds)


@torch.jit.script
def quat_wxyz_to_xyzw(quat_wxyz):
    """
    Convert quaternions from (w, x, y, z) order to (x, y, z, w) order.

    Args:
        quat_wxyz (torch.Tensor): Quaternions in (w, x, y, z) order, shape (..., 4).

    Returns:
        torch.Tensor: Quaternions in (x, y, z, w) order, shape (..., 4).
    """
    inds = torch.tensor([1, 2, 3, 0], dtype=torch.long, device=quat_wxyz.device)
    return torch.index_select(quat_wxyz, dim=-1, index=inds)


@torch.jit.script
def orientation_error_quat(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


@torch.jit.script
def set_goal_position(
    position_limit: torch.Tensor, set_pos: torch.Tensor
) -> torch.Tensor:
    """
    Calculates and returns the desired goal position, clipping the result accordingly to @position_limits.
    @set_pos must be specified to define a global goal position
    """
    # Clip goal position
    set_pos[0] = torch.clamp(set_pos[0], position_limit[0][0], position_limit[0][1])
    set_pos[1] = torch.clamp(set_pos[1], position_limit[1][0], position_limit[1][1])
    set_pos[2] = torch.clamp(set_pos[2], position_limit[2][0], position_limit[2][1])
    return set_pos


@torch.jit.script
def quat2mat(quaternion: torch.Tensor) -> torch.Tensor:
    """Converts given quaternion (x, y, z, w) to matrix.

    Args:
        quaternion: vec4 float angles
    Returns:
        3x3 rotation matrix
    """
    EPS = 1e-8
    inds = torch.tensor([3, 0, 1, 2])
    q = quaternion.clone().detach().float()[inds]

    n = torch.dot(q, q)
    if n < EPS:
        return torch.eye(3)
    q *= math.sqrt(2.0 / n)
    q2 = torch.outer(q, q)
    return torch.tensor(
        [
            [
                1.0 - float(q2[2, 2]) - float(q2[3, 3]),
                float(q2[1, 2]) - float(q2[3, 0]),
                float(q2[1, 3]) + float(q2[2, 0]),
            ],
            [
                float(q2[1, 2]) + float(q2[3, 0]),
                1.0 - float(q2[1, 1]) - float(q2[3, 3]),
                float(q2[2, 3]) - float(q2[1, 0]),
            ],
            [
                float(q2[1, 3]) - float(q2[2, 0]),
                float(q2[2, 3]) + float(q2[1, 0]),
                1.0 - float(q2[1, 1]) - float(q2[2, 2]),
            ],
        ]
    )


# @torch.jit.script
def quat2mat_batched(quaternion: torch.Tensor) -> torch.Tensor:
    """Converts given quaternions (x, y, z, w) to rotation matrices.

    Args:
        quaternion: (..., 4) tensor of quaternions
    Returns:
        (..., 3, 3) tensor of rotation matrices
    """
    EPS = 1e-8
    inds = torch.tensor([3, 0, 1, 2], device=quaternion.device)
    q = quaternion.index_select(-1, inds)

    n = torch.sum(q * q, dim=-1, keepdim=True)
    q *= torch.rsqrt(torch.max(n, torch.tensor(EPS, device=q.device)))

    q1, q2, q3, q0 = torch.unbind(q, dim=-1)
    r11 = 1 - 2 * (q2 * q2 + q3 * q3)
    r12 = 2 * (q1 * q2 - q3 * q0)
    r13 = 2 * (q1 * q3 + q2 * q0)
    r21 = 2 * (q1 * q2 + q3 * q0)
    r22 = 1 - 2 * (q1 * q1 + q3 * q3)
    r23 = 2 * (q2 * q3 - q1 * q0)
    r31 = 2 * (q1 * q3 - q2 * q0)
    r32 = 2 * (q2 * q3 + q1 * q0)
    r33 = 1 - 2 * (q1 * q1 + q2 * q2)

    rot_matrix = torch.stack(
        [
            torch.stack([r11, r12, r13], dim=-1),
            torch.stack([r21, r22, r23], dim=-1),
            torch.stack([r31, r32, r33], dim=-1),
        ],
        dim=-2,
    )

    return rot_matrix


@torch.jit.script
def unit_vector(data: torch.Tensor):
    """Returns ndarray normalized by length, i.e. eucledian norm, along axis."""

    data = torch.clone(data)
    if data.ndim == 1:
        data /= math.sqrt(torch.dot(data, data))
        return data
    length = torch.atleast_1d(torch.sum(data * data))
    length = torch.sqrt(length)
    data /= length
    return data


@torch.jit.script
def quat_multiply(q1: torch.Tensor, q0: torch.Tensor):
    """Return multiplication of two quaternions.
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True
    """
    x0, y0, z0, w0 = float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])
    x1, y1, z1, w1 = float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])
    return torch.tensor(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=torch.float32,
    )


@torch.jit.script
def quat_slerp(
    quat0: torch.Tensor,
    quat1: torch.Tensor,
    fraction: float,
    spin: float = 0,
    shortestpath: bool = True,
):
    """Return spherical linear interpolation between two quaternions.
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True
    >>> q = quat_slerp(q0, q1, 1.0, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True
    """
    EPS = 1e-8
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = torch.dot(q0, q1)
    if abs(abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    d = torch.clip(d, -1.0, 1.0)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


@torch.jit.script
def mat2quat(rmat: torch.Tensor):
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat: 3x3 rotation matrix
    Returns:
        vec4 float quaternion angles
    """
    M = rmat[:3, :3]

    m00 = float(M[0, 0])
    m01 = float(M[0, 1])
    m02 = float(M[0, 2])
    m10 = float(M[1, 0])
    m11 = float(M[1, 1])
    m12 = float(M[1, 2])
    m20 = float(M[2, 0])
    m21 = float(M[2, 1])
    m22 = float(M[2, 2])
    # symmetric matrix K
    K = torch.tensor(
        [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    ).to(rmat.device)
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = torch.linalg.eigh(K)
    inds = torch.tensor([3, 0, 1, 2])
    q1 = V[inds, torch.argmax(w)]
    if q1[0] < 0.0:
        q1 = -q1
    inds = torch.tensor([1, 2, 3, 0])
    return q1[inds]


def matrix_to_quaternion_xyzw(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])

    return torch.stack((o1, o2, o3, o0), -1)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def mat2pose_batched(hmat: torch.Tensor):
    """
    Converts homogeneous 4x4 matrices into poses.

    Args:
        hmat: (..., 4, 4) homogeneous matrices

    Returns:
        tuple (pos, orn) where pos is (..., 3) Cartesian positions
            and orn is (..., 4) quaternions
    """
    pos = hmat[..., :3, 3]
    quat_xyzw = matrix_to_quaternion_xyzw(hmat[..., :3, :3])
    return pos, quat_xyzw


@torch.jit.script
def mat2pose(hmat: torch.Tensor):
    """
    Converts a homogeneous 4x4 matrix into pose.

    Args:
        hmat: a 4x4 homogeneous matrix

    Returns:
        (pos, orn) tuple where pos is vec3 float in cartesian,
            orn is vec4 float quaternion
    """
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn


@torch.jit.script
def set_goal_orientation(set_ori: torch.Tensor):
    """
    Calculates and returns the desired goal orientation, clipping the result accordingly to @orientation_limits.
    @delta and @current_orientation must be specified if a relative goal is requested, else @set_ori must be
    an orientation matrix specified to define a global orientation
    If @axis_angle is set to True, then this assumes the input in axis angle form, that is,
        a scaled axis angle 3-array [ax, ay, az]
    """
    goal_orientation = quat2mat(set_ori)
    return goal_orientation


@torch.jit.script
def pose2mat(
    pos: torch.Tensor, quat: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Converts pose to homogeneous matrix.

    Args:
        pos: a (os, orn tuple where pos is vec3 float cartesian
        quat: orn is vec4 float quaternion.

    Returns:
        4x4 homogeneous matrix
    """
    homo_pose_mat = torch.zeros((4, 4)).to(device)
    homo_pose_mat[:3, :3] = quat2mat(quat)
    homo_pose_mat[:3, 3] = pos
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
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


@torch.jit.script
def pose2mat_batched(
    pos: torch.Tensor, quat_xyzw: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Converts poses to homogeneous matrices.

    Args:
        pos: (B, n_parts, 3) tensor of Cartesian positions
        quat: (B, n_parts, 4) tensor of quaternions

    Returns:
        (B, n_parts, 4, 4) tensor of homogeneous matrices
    """
    B = pos.shape[0]
    n_parts = pos.shape[1]
    homo_pose_mat = torch.zeros((B, n_parts, 4, 4)).to(device)

    homo_pose_mat[:, :, :3, :3] = quaternion_to_matrix(quat_xyzw)
    homo_pose_mat[:, :, :3, 3] = pos
    homo_pose_mat[:, :, 3, 3] = 1.0
    return homo_pose_mat


@torch.jit.script
def to_homogeneous(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Given position and rotation matrix, convert it into homogeneous matrix."""
    transform = torch.zeros((4, 4), device=pos.device)
    if pos.ndim == 2:
        transform[:3, 3:] = pos
    else:
        assert pos.ndim == 1
        transform[:3, 3] = pos
    transform[:3, :3] = rot
    transform[3, 3] = 1

    return transform


@torch.jit.script
def axisangle2quat(vec):
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (torch.Tensor): (ax, ay, az) axis-angle exponential coordinates
    Returns:
        torch.Tensor: (x, y, z, w) vec4 float angles
    """
    # Grab angle
    angle = torch.norm(vec)

    # handle zero-rotation case
    if torch.isclose(angle, torch.tensor([0.0]).to(vec.device)):
        return torch.tensor([0.0, 0.0, 0.0, 1.0]).to(vec.device)

    # make sure that axis is a unit vector
    axis = vec / angle

    q = torch.zeros(4, device=vec.device)
    q[3] = torch.cos(angle / 2.0)
    q[:3] = axis * torch.sin(angle / 2.0)
    return q


@torch.jit.script
def rel_mat(s, t):
    s_inv = torch.linalg.inv(s)
    return t @ s_inv


def rot_mat_tensor(x, y, z, device):
    from furniture_bench.utils.pose import get_mat, rot_mat

    return torch.tensor(rot_mat([x, y, z], hom=True), device=device).float()


@torch.jit.script
def pose_from_vector(vec):
    """
    Vec is (num_envs, 7) tensor where the first 3 elements are the position
    and the last 4 elements are the quaternion x, y, z, w.
    """
    num_envs = vec.shape[0]
    # Extract position and quaternion from the vector
    pos = vec[:, :3]
    quat = vec[:, 3:]

    # Normalize the quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    # Convert quaternion to rotation matrix
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    yy = y * y
    yz = y * z
    yw = y * w
    zz = z * z
    zw = z * w

    rot_matrix = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            1 - 2 * (xx + zz),
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).view(num_envs, 3, 3)

    # Combine position and rotation matrix to form the pose matrix
    pose_matrix = torch.eye(4, dtype=vec.dtype, device=vec.device).repeat(
        num_envs, 1, 1
    )
    pose_matrix[:, :3, :3] = rot_matrix
    pose_matrix[:, :3, 3] = pos

    return pose_matrix


def cosine_sim(w, v):
    # Compute the dot product and norms along the last dimension
    dot_product = torch.sum(w * v, dim=-1)
    w_norm = torch.norm(w, dim=-1)
    v_norm = torch.norm(v, dim=-1)

    # Compute the cosine similarity
    return dot_product / (w_norm * v_norm)


@torch.jit.script
def is_similar_rot(rot1: torch.Tensor, rot2: torch.Tensor, ori_bound: float):
    cosine_sims = cosine_sim(rot1, rot2)
    return torch.all(cosine_sims >= ori_bound, dim=-1)


@torch.jit.script
def is_similar_pos(pos1: torch.Tensor, pos2: torch.Tensor, pos_threshold: torch.Tensor):
    pos_diffs = torch.abs(pos1 - pos2)
    within_threshold = pos_diffs <= pos_threshold
    return torch.all(within_threshold, dim=-1)
