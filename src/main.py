"""Run the Sudoku vision and solving pipeline."""

import time

from solver.solver import solve_sudoku
from vision.vis_board import capture_image
from vision.vision import run_vision_pipeline


def main() -> None:
    """Capture a Sudoku board, recognise it, and solve it."""
    image = capture_image()
    sudoku, _ = run_vision_pipeline(image)
    solve_start = time.perf_counter()
    solve_sudoku(
        sudoku,
        print_unsolved=True,
        print_solved=True
    )
    print(
        f"sudoku solved in {time.perf_counter() - solve_start:.3f}s",
        flush=True
    )


if __name__ == "__main__":
    main()
