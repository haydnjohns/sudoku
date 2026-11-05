
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

DEMO_IMAGE_PATH = ASSETS_DIR / "demo.png"


# --------------------------------------------------------------------------
# Camera calibration
# --------------------------------------------------------------------------

CAMERA_MATRIX = np.array([
    [2650.9964176101485, 0.0, 1638.4541745042789],
    [0.0, 2644.434621507731, 1106.6683119862594],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DISTORTION_COEFFICIENTS = np.array([
    0.19930490026457645,
    -0.2903695599399088,
    -0.007406225233953261,
    0.005002318775684634,
    -0.5143858238250185
], dtype=np.float32)


# --------------------------------------------------------------------------
# Camera to robot transformation
# --------------------------------------------------------------------------

# 4x4 homogeneous transformation matrix.
# Converts coordinates from the camera frame to the robot frame.
# Replace with your calibrated transformation.

ROBOT_T_CAMERA = np.eye(4, dtype=np.float64)


# --------------------------------------------------------------------------
# Sudoku board dimensions
# --------------------------------------------------------------------------

BOARD_WIDTH_MM = 180.0
BOARD_HEIGHT_MM = 180.0

BOARD_IMAGE_SIZE = 900
GRID_SIZE = 9


# --------------------------------------------------------------------------
# Image processing
# --------------------------------------------------------------------------

GAUSSIAN_KERNEL = (9, 9)

ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11
ADAPTIVE_THRESHOLD_C = 2

MORPHOLOGY_KERNEL_SIZE = 3


# --------------------------------------------------------------------------
# Digit recognition
# --------------------------------------------------------------------------
