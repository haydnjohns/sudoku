
from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class BoardPose:
    position_mm: np.ndarray
    rotation_matrix: np.ndarray
    transformation_matrix: np.ndarray


# --------------------------------------------------------------------------
# Board pose estimation
# --------------------------------------------------------------------------

def calculate_board_pose(corners: np.ndarray) -> BoardPose:
    """
    Calculate the Sudoku board pose in robot coordinates.

    Corners must be undistorted pixel coordinates ordered:
    top-left, top-right, bottom-right, bottom-left.

    Physical dimensions are in millimetres.
    """

    # Physical coordinates of board corners.
    object_points = np.array(
        [
            [0, 0, 0],
            [config.BOARD_WIDTH_MM, 0, 0],
            [config.BOARD_WIDTH_MM, config.BOARD_HEIGHT_MM, 0],
            [0, config.BOARD_HEIGHT_MM, 0]
        ],
        dtype=np.float32
    )

    image_points = corners.astype(np.float32)

    # The image points have already been undistorted.
    zero_distortion = np.zeros_like(
        config.DISTORTION_COEFFICIENTS
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        config.CAMERA_MATRIX,
        zero_distortion
    )

    if not success:
        raise ValueError("Could not calculate board pose")

    rotation_camera_board, _ = cv2.Rodrigues(rotation_vector)

    # Transform from board coordinates to camera coordinates.
    camera_T_board = np.eye(4)

    camera_T_board[:3, :3] = rotation_camera_board
    camera_T_board[:3, 3] = translation_vector.flatten()

    # Transform from board coordinates to robot coordinates.
    robot_T_board = config.ROBOT_T_CAMERA @ camera_T_board

    return BoardPose(
        position_mm=robot_T_board[:3, 3],
        rotation_matrix=robot_T_board[:3, :3],
        transformation_matrix=robot_T_board
    )


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from vis_board import capture_image, process_image

    image = capture_image()
    board = process_image(image)

    pose = calculate_board_pose(board.corners)

    print("Board position in robot coordinates (mm):")
    print(pose.position_mm)

    print("Board rotation:")
    print(pose.rotation_matrix)
