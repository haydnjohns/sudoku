"""Capture, detect and rectify a Sudoku board, and locate the camera on it.

Board frame (the frame the camera pose is reported in, all in mm):
    origin  bottom-left corner of the board, on the board surface
    x       along the bottom edge, to the right
    y       up the left edge, towards the top of the board
    z       up, out of the board

A 200 x 200 mm board therefore spans (0, 0) to (200, 200), and a camera
mounted just below the board's centre sits near (100, -75, CAMERA_HEIGHT).

Camera axes (used internally while measuring the board):
    origin  the point on the board plane directly below the camera
    x       the camera's right
    y       the camera's forward direction, projected onto the board plane
"""

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
class CameraPose:
    """Camera pose in the board frame, plus the measured board size.

    x, y: stamper position, i.e. the camera position plus
        CAMERA_TO_STAMPER_OFFSET_MM along the camera's x axis (mm).
    theta: angle of the camera's x axis from the board's x axis, degrees CCW.
    board_width, board_height: measured board size (mm).
    """

    x: float
    y: float
    theta: float
    dx_top: float
    dx_btm: float
    dy_left: float
    dy_top: float

    @property
    def board_width(self) -> float:
        return (self.dx_top + self.dx_btm) / 2

    @property
    def board_height(self) -> float:
        return (self.dy_left + self.dy_top) / 2


BoardPose = CameraPose


@dataclass
class RectifiedBoard:
    image: np.ndarray
    corners: np.ndarray
    transformation_matrix: np.ndarray
    camera_matrix: np.ndarray
    camera_pose: CameraPose

    @property
    def pose(self) -> CameraPose:
        return self.camera_pose


def capture_image(width: int = 2592, height: int = 1944) -> np.ndarray:
    """Capture an image using the Raspberry Pi camera."""
    start = time.perf_counter()
    process = subprocess.Popen(
        ["rpicam-jpeg", "-o", "-", "--width", str(width), "--height", str(height),
         "--nopreview"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(f"Camera launched in {time.perf_counter() - start:.3f}s", flush=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, process.args, output=stdout, stderr=stderr
        )

    image = cv2.imdecode(np.frombuffer(stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode captured image")

    print(f"Image captured in {time.perf_counter() - start:.3f}s", flush=True)
    return image


def _calculate_camera_pose_and_corners(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[CameraPose, np.ndarray]:
    """Locate the camera/stamper in the board frame from the board's pixel corners.

    corners: (4, 2) pixel coordinates in the undistorted, upright image, ordered
        top-left, top-right, bottom-right, bottom-left.
    camera_matrix: intrinsics that match those pixel coordinates (already
        adjusted for any 180-degree image rotation).
    """
    if corners.shape != (4, 2):
        raise ValueError("Expected four pixel corners with shape (4, 2)")
    if config.CAMERA_HEIGHT <= 0:
        raise ValueError("CAMERA_HEIGHT must be positive")
    if not 0 < config.CAMERA_ANGLE <= 90:
        raise ValueError("CAMERA_ANGLE must be in the range (0, 90]")

    # Back-project each corner to a ray in the camera frame (x right, y down,
    # z along the optical axis).
    pixels = np.column_stack((corners.astype(np.float64), np.ones(4)))
    rays = (np.linalg.inv(camera_matrix) @ pixels.T).T

    # Undo the camera's downward tilt so each ray has a horizontal "forward"
    # component and a vertical "up" component. Corners must point below the
    # horizon to hit the board plane.
    tilt = np.deg2rad(config.CAMERA_ANGLE)
    forward = np.cos(tilt) * rays[:, 2] - np.sin(tilt) * rays[:, 1]
    up = -np.sin(tilt) * rays[:, 2] - np.cos(tilt) * rays[:, 1]
    if np.any(up >= 0):
        raise ValueError("Board corners do not project in front of the board plane")

    # Intersect the rays with the board plane, CAMERA_HEIGHT below the camera.
    # This gives each corner's position on the plane in camera axes (mm):
    # origin directly below the camera, x to its right, y forward.
    scale = -config.CAMERA_HEIGHT / up
    corners_cam = np.column_stack((rays[:, 0] * scale, forward * scale))

    # Use the measured projected corners directly. The board origin is the
    # measured bottom-left corner, and each board axis follows its measured
    # edge. No rectangle, right-angle, or equal-edge constraint is imposed.
    board_origin = corners_cam[3]
    bottom_edge = corners_cam[2] - corners_cam[3]
    left_edge = corners_cam[0] - corners_cam[3]
    width_bottom = np.linalg.norm(bottom_edge)
    width_top = np.linalg.norm(corners_cam[1] - corners_cam[0])
    height_left = np.linalg.norm(left_edge)
    height_right = np.linalg.norm(corners_cam[2] - corners_cam[1])
    if min(width_bottom, width_top, height_left, height_right) <= 0:
        raise ValueError("Detected board has invalid dimensions")
    board_x = bottom_edge / width_bottom
    board_y = left_edge / height_left
    board_angle = np.arctan2(board_x[1], board_x[0])
    measured_global_corners = np.column_stack((
        (corners_cam - board_origin) @ board_x,
        (corners_cam - board_origin) @ board_y,
        np.zeros(4),
    ))
    print("Original board corners in the global frame (mm, TL/TR/BR/BL):")
    for index, point in enumerate(measured_global_corners):
        print(f"  {index}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")

    # Change of frame. The camera is at (0, 0) in camera axes, so the vector
    # from the board origin to the camera is (0 - board_origin). Resolving it
    # along the board axes gives the camera's position in the board frame.
    camera_x = -board_origin @ board_x
    camera_y = -board_origin @ board_y

    # The camera's x axis sits at -board_angle from the board's x axis. The
    # stamper is offset from the camera along that axis.
    theta = -np.rad2deg(board_angle)
    offset = config.CAMERA_TO_STAMPER_OFFSET_MM
    return CameraPose(
        x=float(camera_x + offset * np.cos(np.deg2rad(theta))),
        y=float(camera_y + offset * np.sin(np.deg2rad(theta))),
        theta=float(theta),
        dx_top=float(width_top),
        dx_btm=float(width_bottom),
        dy_left=float(height_left),
        dy_top=float(height_right),
    ), corners_cam


def calculate_camera_pose(corners: np.ndarray, camera_matrix: np.ndarray) -> CameraPose:
    """Locate the camera/stamper in the board frame from board corners."""
    pose, _ = _calculate_camera_pose_and_corners(corners, camera_matrix)
    return pose


def process_image(image: np.ndarray, debug: bool = False) -> RectifiedBoard:
    """Undistort, detect and rectify a Sudoku board, and calculate the camera pose.

    Returned corners are pixel coordinates in the upright, undistorted image.
    """
    height, width = image.shape[:2]

    # --- Undistort to a pinhole model --------------------------------------
    # Grayscale first: it's all the later steps need, and undistorting one
    # channel is roughly three times cheaper than three.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        config.CAMERA_MATRIX, config.DISTORTION_COEFFICIENTS, (width, height), alpha=0
    )
    gray = cv2.undistort(
        gray, config.CAMERA_MATRIX, config.DISTORTION_COEFFICIENTS, None, camera_matrix
    )

    # --- Make the image upright --------------------------------------------
    # A 180-degree rotation maps pixel (u, v) to (W-1-u, H-1-v), so the
    # principal point has to move with it. This is the only place it's done.
    if config.CAMERA_MOUNTED_UPSIDE_DOWN:
        gray = cv2.rotate(gray, cv2.ROTATE_180)
        camera_matrix[0, 2] = width - 1 - camera_matrix[0, 2]
        camera_matrix[1, 2] = height - 1 - camera_matrix[1, 2]

    # --- Binary image for board detection ----------------------------------
    blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_KERNEL, 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        config.ADAPTIVE_THRESHOLD_BLOCK_SIZE, config.ADAPTIVE_THRESHOLD_C,
    )
    binary = cv2.bitwise_not(binary)  # lines/ink become white
    # Dilate then erode (a "close") to join broken grid lines.
    kernel = np.ones((config.MORPHOLOGY_KERNEL_SIZE,) * 2, np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # --- Find the board: largest contour that simplifies to four points ----
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        approximation = cv2.approxPolyDP(
            contour, 0.02 * cv2.arcLength(contour, True), True
        )
        if len(approximation) == 4:
            break
    else:
        raise ValueError("Could not find Sudoku board")

    # Order corners TL, TR, BR, BL: the two highest points in the image are the
    # top edge and the two lowest are the bottom edge, then split each pair
    # left/right by x. Unlike ordering by x+y / y-x, this holds up when
    # perspective makes the near edge much wider than the far edge. It assumes
    # the board is within about 45 degrees of upright in the image.
    points = approximation.reshape(4, 2).astype(np.float32)
    by_y = points[np.argsort(points[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    corners = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

    # --- Camera pose in the board frame ------------------------------------
    camera_pose, _ = _calculate_camera_pose_and_corners(
        corners,
        camera_matrix,
    )

    # --- Perspective-correct to a square, top-down grayscale image ---------
    size = config.BOARD_IMAGE_SIZE
    destination = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(gray, matrix, (size, size))

    if debug:
        debug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        measured_polygon = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            debug_image,
            [measured_polygon],
            True,
            (0, 0, 255),
            3,
        )
        for index, point in enumerate(corners.astype(int)):
            cv2.circle(debug_image, tuple(point), 8, (0, 0, 255), -1)
            cv2.putText(
                debug_image,
                f"M{index}",
                tuple(point + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        cv2.imshow("Undistorted board corners", debug_image)

    return RectifiedBoard(
        image=warped,
        corners=corners,
        transformation_matrix=matrix,
        camera_matrix=camera_matrix,
        camera_pose=camera_pose,
    )


if __name__ == "__main__":
    image = cv2.imread(str(config.DEMO_IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not load demo image: {config.DEMO_IMAGE_PATH}")
    board = process_image(image, debug=True)
    print("Undistorted board corners:")
    print(board.corners)
    print("Calculated camera pose in the board frame:")
    print(board.camera_pose)
    cv2.waitKey(0)
    cv2.destroyAllWindows()