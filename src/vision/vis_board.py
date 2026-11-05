
"""Capture, preprocess, detect, and perspective-correct a Sudoku board image."""

from dataclasses import dataclass
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


@dataclass
class RectifiedBoard:
    image: np.ndarray
    corners: np.ndarray
    transformation_matrix: np.ndarray


# --------------------------------------------------------------------------
# Image capture
# --------------------------------------------------------------------------

def capture_image(width: int = 2592, height: int = 1944) -> np.ndarray:
    """Capture an image using the Raspberry Pi camera."""
    camera_start = time.perf_counter()
    process = subprocess.Popen(
        [
            "rpi-cam",
            "-o", "-",
            "--width", str(width),
            "--height", str(height),
            "--nopreview",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(
        f"Camera launched in {time.perf_counter() - camera_start:.3f}s",
        flush=True
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            process.args,
            output=stdout,
            stderr=stderr
        )

    image = cv2.imdecode(
        np.frombuffer(stdout, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    if image is None:
        raise RuntimeError("Failed to decode captured image")

    image = cv2.rotate(image, cv2.ROTATE_180)
    print(
        f"Image captured in {time.perf_counter() - camera_start:.3f}s",
        flush=True
    )
    return image


# --------------------------------------------------------------------------
# Image undistortion
# --------------------------------------------------------------------------

def undistort_image(image: np.ndarray) -> np.ndarray:
    """
    Convert the image to the calibrated pinhole-camera model.

    The original camera matrix is retained as the new camera matrix,
    so the resulting pixel coordinates use config.CAMERA_MATRIX.
    """
    return cv2.undistort(
        image,
        config.CAMERA_MATRIX,
        config.DISTORTION_COEFFICIENTS,
        None,
        config.CAMERA_MATRIX
    )


# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Create a binary image suitable for board detection."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        grayscale,
        config.GAUSSIAN_KERNEL,
        0
    )

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config.ADAPTIVE_THRESHOLD_BLOCK_SIZE,
        config.ADAPTIVE_THRESHOLD_C
    )

    binary = cv2.bitwise_not(binary)

    kernel = np.ones(
        (config.MORPHOLOGY_KERNEL_SIZE,) * 2,
        np.uint8
    )

    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    return binary


# --------------------------------------------------------------------------
# Board detection
# --------------------------------------------------------------------------

def find_board_contour(binary_image: np.ndarray) -> np.ndarray:
    """Find the largest four-sided contour."""
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approximation) == 4:
            return approximation

    raise ValueError("Could not find Sudoku board")


def sort_corners(contour: np.ndarray) -> np.ndarray:
    """
    Order corners as:
    top-left, top-right, bottom-right, bottom-left.
    """
    points = contour.reshape(4, 2).astype(np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).flatten()

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]

    return ordered


# --------------------------------------------------------------------------
# Perspective correction
# --------------------------------------------------------------------------

def rectify_board(
    image: np.ndarray,
    corners: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the board into a square, top-down grayscale image."""
    size = config.BOARD_IMAGE_SIZE

    destination = np.array(
        [
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        corners,
        destination
    )

    grayscale = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    warped = cv2.warpPerspective(
        grayscale,
        matrix,
        (size, size)
    )

    return warped, matrix


# --------------------------------------------------------------------------
# Main processing function
# --------------------------------------------------------------------------

def process_image(
    image: np.ndarray,
    debug: bool = False
) -> RectifiedBoard:
    """
    Undistort, detect and rectify a Sudoku board.

    Returned corners are pixel coordinates in the undistorted image.
    """
    undistorted = undistort_image(image)

    binary = preprocess_image(undistorted)

    contour = find_board_contour(binary)
    corners = sort_corners(contour)

    warped, matrix = rectify_board(
        undistorted,
        corners
    )

    if debug:
        cv2.imshow("Original", image)
        cv2.imshow("Undistorted", undistorted)
        cv2.imshow("Binary", binary)
        cv2.imshow("Rectified Board", warped)

    return RectifiedBoard(
        image=warped,
        corners=corners,
        transformation_matrix=matrix
    )


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    image = cv2.imread(str(config.DEMO_IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not load demo image: {config.DEMO_IMAGE_PATH}")
    board = process_image(image, debug=True)
    print("Undistorted board corners:")
    print(board.corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
