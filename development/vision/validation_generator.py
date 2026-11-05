"""Generate labelled real OCR cells after manual visual confirmation."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from src.vision.vis_board import capture_image, process_image
from src.vision.vis_cells import extract_cell, preprocess_digit


EXPECTED = "9456732152473556498732481356"
REAL_CELLS_ROOT = REPO_ROOT / "assets" / "vision" / "real_cells"
DEFAULT_IMAGE = REPO_ROOT / "assets" / "vision" / "test grids" / "ss" / "camera_test_good.jpg"


def parse_args() -> argparse.Namespace:
    """Parse image-source options."""
    parser = argparse.ArgumentParser(
        description="Generate manually confirmed real OCR validation cells."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Path to a board image.",
    )
    source.add_argument(
        "--capture",
        action="store_true",
        help="Capture a board image using the vision camera module.",
    )
    return parser.parse_args()


def load_image(args: argparse.Namespace):
    """Load an image from disk or capture one from the camera."""
    if args.capture:
        return capture_image()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    return image


def collect_cells(board_image):
    """Collect detected digit canvases and their source cell coordinates."""
    cells = []
    for row in range(9):
        for col in range(9):
            cell = extract_cell(board_image, row, col)
            canvas = preprocess_digit(cell)
            if canvas is not None:
                cells.append((canvas, row, col))
    return cells


def confirm_cells(cells) -> bool:
    """Show each labelled canvas and return whether all were confirmed."""
    for index, (canvas, _, _) in enumerate(cells):
        label = EXPECTED[index]
        cv2.imshow(str(label), canvas)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(str(label))
        if key == ord("q"):
            return False
    return True


def save_cells(cells) -> None:
    """Save confirmed canvases under their hard-coded labels."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_counts = {digit: 0 for digit in set(EXPECTED)}

    for index, (canvas, row, col) in enumerate(cells):
        label = EXPECTED[index]
        output_dir = REAL_CELLS_ROOT / label
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{timestamp}_r{row}c{col}.png"

        if not cv2.imwrite(str(output_path), canvas):
            raise RuntimeError(f"Failed to save {output_path}")
        saved_counts[label] += 1

    for digit in sorted(saved_counts):
        print(f"Saved {saved_counts[digit]} cells for digit {digit}")


def main() -> None:
    """Build and manually confirm a labelled real-cell dataset."""
    args = parse_args()
    image = load_image(args)
    board = process_image(image)
    cells = collect_cells(board.image)

    if len(cells) != len(EXPECTED):
        print(
            f"Detected {len(cells)} cells; expected {len(EXPECTED)}. "
            "Nothing was saved."
        )
        return

    try:
        confirmed = confirm_cells(cells)
    finally:
        cv2.destroyAllWindows()

    if not confirmed:
        print("Cell confirmation aborted. Nothing was saved.")
        return

    save_cells(cells)


if __name__ == "__main__":
    main()
