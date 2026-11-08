"""Orchestrate the image-processing and digit-recognition pipeline."""

import cv2
import numpy as np
import time

try:
    import config
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config

from solver.solver import print_sudoku

try:
    from .vis_board import process_image
    from .vis_cells import recognise_board
    from .vis_board import BoardPose
except ImportError:
    from vis_board import process_image
    from vis_cells import recognise_board
    from vis_board import BoardPose


def print_board_pose(pose: BoardPose) -> None:
    """Print the calculated stamper position and board dimensions."""
    print(
        "Board geometry (stamper position, board frame):\n"
        f"  x: {pose.x:.2f} mm\n"
        f"  y: {pose.y:.2f} mm\n"
        f"  theta: {pose.theta:.2f} degrees\n"
        f"  dx_top: {pose.dx_top:.2f} mm\n"
        f"  dx_btm: {pose.dx_btm:.2f} mm\n"
        f"  dy_left: {pose.dy_left:.2f} mm\n"
        f"  dy_top: {pose.dy_top:.2f} mm",
        flush=True,
    )


def run_vision_pipeline(
    image: np.ndarray
) -> tuple[np.ndarray, BoardPose, np.ndarray]:
    """Detect a board, recognise its digits, pose, and prediction vectors."""
    rectified_board = process_image(image)
    ocr_start = time.perf_counter()
    sudoku, prediction_vectors = recognise_board(rectified_board.image)
    print(
        f"OCR complete in {time.perf_counter() - ocr_start:.3f}s",
        flush=True
    )

    return sudoku, rectified_board.pose, prediction_vectors


def main() -> None:
    """Run the vision demo using the configured demo image."""
    image = cv2.imread(str(config.DEMO_IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(
            f"Could not load demo image: {config.DEMO_IMAGE_PATH}"
        )

    sudoku, pose, _ = run_vision_pipeline(image)
    print_board_pose(pose)
    print_sudoku(sudoku)


if __name__ == "__main__":
    main()
