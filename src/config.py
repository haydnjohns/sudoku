
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
VISION_ASSETS_DIR = ASSETS_DIR / "vision"

DEMO_IMAGE_PATH = VISION_ASSETS_DIR / "test grids" / "weblarge.jpg"
VISION_MODEL_PATH = VISION_ASSETS_DIR / "models" / "vis_model_new.tflite"


# --------------------------------------------------------------------------
# Camera calibration
# --------------------------------------------------------------------------

CAMERA_MATRIX = np.array([
    [2650.9964176101485, 0.0, 1638.4541745042789],
    [0.0, 2644.434621507731, 1106.6683119862594],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Set this to False if the camera is mounted in its normal orientation.
CAMERA_MOUNTED_UPSIDE_DOWN = True

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

CAMERA_HEIGHT = 73  # Camera pose relative to the board plane.
CAMERA_ANGLE = 29   # Degrees down from horizontal; the geometry module uses this convention.

BOARD_IMAGE_SIZE = 900
GRID_SIZE = 9

# Robot geometry and initial board-relative position.
STAMPER_SETBACK = 16.0
CAMERA_TO_STAMPER_OFFSET_MM = -10.0
ROW_TRAVEL_TOLERANCE = 5.0
MOVEMENT_WAIT_SECONDS = 0.1
STAMP_WAIT_SECONDS = 1.0


# --------------------------------------------------------------------------
# Image processing
# --------------------------------------------------------------------------

GAUSSIAN_KERNEL = (9, 9)
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11
ADAPTIVE_THRESHOLD_C = 2
MORPHOLOGY_KERNEL_SIZE = 3

