import numpy as np


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14
    # yaw = np.array(yaw)
    # pitch = np.array(pitch)
    # roll = np.array(roll)

    pitch_mat = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            np.cos(pitch),
            -np.sin(pitch),
            0.0,
            np.sin(pitch),
            np.cos(pitch),
        ]
    )
    pitch_mat = pitch_mat.reshape(3, 3)

    yaw_mat = np.array(
        [
            np.cos(yaw),
            0.0,
            np.sin(yaw),
            0.0,
            1.0,
            0.0,
            -np.sin(yaw),
            0.0,
            np.cos(yaw),
        ],
    )
    yaw_mat = yaw_mat.reshape(3, 3)

    roll_mat = np.array(
        [
            np.cos(roll),
            -np.sin(roll),
            0.0,
            np.sin(roll),
            np.cos(roll),
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    )
    roll_mat = roll_mat.reshape(3, 3)

    rot_mat = np.einsum("ij,jk,km->im", pitch_mat, yaw_mat, roll_mat)

    return rot_mat
