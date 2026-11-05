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
    from .vis_geometry import BoardPose, calculate_board_pose
except ImportError:
    from vis_board import process_image
    from vis_cells import recognise_board
    from vis_geometry import BoardPose, calculate_board_pose


def run_vision_pipeline(image: np.ndarray) -> tuple[np.ndarray, BoardPose]:
    """Detect a board, recognise its digits, and estimate its pose."""
    rectified_board = process_image(image)
    pose = calculate_board_pose(rectified_board.corners)
    ocr_start = time.perf_counter()
    sudoku = recognise_board(rectified_board.image)
    print(
        f"OCR complete in {time.perf_counter() - ocr_start:.3f}s",
        flush=True
    )

    return sudoku, pose


def main() -> None:
    """Run the vision demo using the configured demo image."""
    image = cv2.imread(str(config.DEMO_IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(
            f"Could not load demo image: {config.DEMO_IMAGE_PATH}"
        )

    sudoku, _ = run_vision_pipeline(image)
    print_sudoku(sudoku)


if __name__ == "__main__":
    main()
