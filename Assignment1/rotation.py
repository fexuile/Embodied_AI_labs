import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Unnormalized quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        Normalized quaternion with shape (4,)
    """
    return q / np.linalg.norm(q)
    # raise NotImplementedError("Implement this function")


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Return the conjugate of the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The conjugate of the quaternion with shape (4,)
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])
    # raise NotImplementedError("Implement this function")


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply the two quaternions.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    np.ndarray
        The multiplication result with shape (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])
    # raise NotImplementedError("Implement this function")


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Use quaternion to rotate a 3D vector.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)
    v: np.ndarray
        Vector with shape (3,)

    Returns
    -------
    np.ndarray
        The rotated vector with shape (3,)
    """
    q_v = np.concatenate(([0], v))
    q_conj = quat_conjugate(q)
    return quat_multiply(quat_multiply(q, q_v), q_conj)[1:]
    # raise NotImplementedError("Implement this function")


def quat_relative_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute the relative rotation angle between the two quaternions.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    float
        The relative rotation angle in radians, greater than or equal to 0.
    """
    return 2 * np.arccos(np.abs(np.dot(q1, q2)))
    # raise NotImplementedError("Implement this function")


def interpolate_quat(q1: np.ndarray, q2: np.ndarray, ratio: float) -> np.ndarray:
    """
    Interpolate between two quaternions with given ratio.

    When the ratio is 0, return q1; when the ratio is 1, return q2.

    The interpolation should be done in the shortest minor arc connecting the quaternions on the unit sphere.

    If there are multiple correct answers, you can output any of them.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)
    ratio: float
        The ratio of interpolation, should be in [0, 1]

    Returns
    -------
    np.ndarray
        The interpolated quaternion with shape (4,)

    Note
    ----
    What should be done if the inner product of the quaternions is negative?
    """
    if np.dot(q1, q2) < 0:
        q1 = -q1
    phi = np.arccos(np.dot(q1, q2))
    return q1 * np.sin((1   - ratio) * phi) / np.sin(phi) + q2 * np.sin(ratio * phi) / np.sin(phi)
    # raise NotImplementedError("Implement this function")


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to rotation matrix.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])
    # raise NotImplementedError("Implement this function")


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to quaternion.

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """
    m00, m01, m02 = mat[0]
    m10, m11, m12 = mat[1]
    m20, m21, m22 = mat[2]
    tr = m00 + m11 + m22
    w = np.sqrt(tr + 1) / 2
    x = (m21 - m12) / (4 * w)
    y = (m02 - m20) / (4 * w)
    z = (m10 - m01) / (4 * w)
    return np.array([w, x, y, z])
    # raise NotImplementedError("Implement this function")


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi.

    If there are multiple answers, you can output any.

    Parameters
    ----------
    q: np.ndarray
        The quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """
    w, x, y, z = q
    angle = 2 * np.arccos(w)
    mul = 1
    if angle > np.pi:
        angle = 2 * np.pi - angle
        mul = -1
    if angle == 0:
        return np.zeros(3)
    axis = np.array([x, y, z]) / np.sin(angle / 2)
    # print(w, x, y, z)
    # print(angle * axis)
    return angle * axis * mul
    # raise NotImplementedError("Implement this function")


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to quaternion.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """
    angle = np.linalg.norm(aa)
    w = np.cos(angle / 2)
    axis = aa / angle * np.sin(angle / 2)
    return np.array([w, *axis])
    # raise NotImplementedError("Implement this function")


def axis_angle_to_mat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to rotation matrix.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    return quat_to_mat(axis_angle_to_quat(aa))


def mat_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """
    return quat_to_axis_angle(mat_to_quat(mat))


def uniform_random_quat() -> np.ndarray:
    """
    Generate a random quaternion with uniform distribution.

    Returns
    -------
    np.ndarray
        The random quaternion with shape (4,)
    """
    q = quat_normalize(np.random.normal(0, 1, size=(4, )))
    return q
    # raise NotImplementedError("Implement this function")


def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    """
    Convert roll-pitch-yaw euler angles into rotation matrix.

    This is required since URDF use this as rotation representation.

    Parameters
    ----------
    rpy: np.ndarray
        The euler angles with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    roll, pitch, yaw = rpy

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = R_z @ R_y @ R_x  # Matrix multiplication in ZYX order
    return R
