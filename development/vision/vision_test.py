"""Evaluate the complete Sudoku vision pipeline against labelled test images.

This file is intentionally self-contained.  It can be copied while tuning the
vision pipeline without changing the application modules:

    python development/vision/vision_test.py

The test images and labels are the current fixtures in
``assets/vision/test grids/ss``.  A non-zero exit status means that an image
could not be processed, not that the model made an OCR error; OCR errors are
reported in the per-image summary so experiments remain useful.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from skimage.segmentation import clear_border


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = ROOT / "assets" / "vision" / "test grids"
DEFAULT_MODEL = ROOT / "assets" / "vision" / "models" / "vis_model_new.tflite"

# Set to 0 to keep debug output from previous runs.
delete_debug = 1

# Comment out any grids that should be skipped during the default run.
SELECTED_IMAGES = [
    "camera_test_bad.jpg",
    "camera_test_good.jpg",
    "demo.png",
    "grid6.jpg",
    "grid7.jpg",
    "grid8.jpg",
    "grid9.jpg",
    "krazydad.jpg",
    "thicc.jpg",
    "weblarge.jpg",
   "websmall.jpg",
]

CAMERA_MATRIX = np.array(
    [[2650.9964, 0.0, 1638.4542], [0.0, 2644.4346, 1106.6683], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
DISTORTION_COEFFICIENTS = np.array(
    [0.1993049, -0.29036956, -0.007406225, 0.005002319, -0.51438582],
    dtype=np.float32,
)
BOARD_SIZE = 900
GRID_SIZE = 9
CELL_SIZE = BOARD_SIZE // GRID_SIZE
CELL_MARGIN_RATIO = 0.25
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2
GAUSSIAN_KERNEL = (9, 9)
MORPHOLOGY_KERNEL_SIZE = 3
MIN_CONTOUR_AREA = 300
DIGIT_SIZE = 20
CANVAS_SIZE = 28
MIN_PREDICTION_CONFIDENCE = 0.5
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


# Rows are kept as strings so the expected grid is easy to compare with the
# solutions in TODO.md.  A period denotes an empty cell.
BASELINES: dict[str, tuple[str, ...]] = {
    "camera_test_bad.jpg": (
        "94.5..6.7",
        ".3......2",
        "1.52.....",
        "4.7.....5",
        ".........",
        "5....64.9",
        ".....87.3",
        "2......4.",
        "8.1..3.56",
    ),
    "camera_test_good.jpg": (
        "94.5..6.7",
        ".3......2",
        "1.52.....",
        "4.73....5",
        ".........",
        "5....64.9",
        ".....87.3",
        "2......4.",
        "8.1..3.56",
    ),
    "demo.png": (
        ".....4..6",
        "..25..74.",
        ".1....82.",
        ".2..1...7",
        "...278...",
        "9...5..8.",
        ".83....9.",
        ".47..53..",
        "2..6.....",
    ),
    "grid6.jpg": (
        "555555555",
        "666666666",
        "777777777",
        "888888888",
        "999999999",
        "111111111",
        "222222222",
        "333333333",
        "444444444",
    ),
    "grid7.jpg": (
        "444444444",
        "555555555",
        "666666666",
        "777777777",
        "888888888",
        "999999999",
        "111111111",
        "222222222",
        "333333333",
    ),
    "grid8.jpg": (
        "333333333",
        "444444444",
        "555555555",
        "666666666",
        "777777777",
        "888888888",
        "999999999",
        "111111111",
        "222222222",
    ),
    "grid9.jpg": (
        "222222222",
        "333333333",
        "444444444",
        "555555555",
        "666666666",
        "777777777",
        "888888888",
        "999999999",
        "111111111",
    ),
    "krazydad.jpg": (
        "2.5..7..6",
        "4..96..2.",
        "....8..45",
        "98..74...",
        "57.8.2.69",
        "...63..57",
        "7.5..2....",
        ".6..51..2",
        "3..4..5.8",
    ),
    "thicc.jpg": (
        "5..8.9...",
        "......721",
        "7.3....5.",
        "624......",
        "...3..24.",
        ".5.6...9.",
        "96.......",
        "......8.4",
        "....57...",
    ),
    "weblarge.jpg": (
        "...7.41..",
        "76..12.43",
        "..23...8.",
        "...625...",
        "..6...4..",
        "...843...",
        ".2...18..",
        "91.46..27",
        "..72.9...",
    ),
    "websmall.jpg": (
        "..3.8....",
        "15.492...",
        "..975.4..",
        "547....6.",
        "8..267..5",
        ".2....781",
        "..2.781..",
        "...134.96",
        "....2.8..",
    )
}


@dataclass
class CellResult:
    row: int
    col: int
    expected: int
    extracted: bool
    predicted: int | None = None
    confidence: float = 0.0
    contour_area: float | None = None
    contour_bounding_box_area: int | None = None
    reason: str = ""
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class BoardDetection:
    corners: np.ndarray
    contour: np.ndarray
    contour_area: float


@dataclass
class ImageResult:
    path: Path
    cells: list[CellResult] = field(default_factory=list)
    board_contour_area: float | None = None
    board_contour_image: np.ndarray | None = None
    rectified_image: np.ndarray | None = None
    error: str | None = None

    @property
    def failures(self) -> list[CellResult]:
        return [cell for cell in self.cells if cell.reason]


def expected_grid(path: Path) -> np.ndarray:
    """Return a labelled 9x9 grid for a fixture filename."""
    try:
        rows = BASELINES[path.name]
    except KeyError as exc:
        raise ValueError(f"No baseline is defined for {path.name}") from exc
    return np.array(
        [[0 if value == "." else int(value) for value in row] for row in rows],
        dtype=np.int32,
    )


def undistort_image(image: np.ndarray) -> np.ndarray:
    return cv2.undistort(
        image, CAMERA_MATRIX, DISTORTION_COEFFICIENTS, None, CAMERA_MATRIX
    )


def preprocess_board(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, GAUSSIAN_KERNEL, 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    )
    kernel = np.ones((MORPHOLOGY_KERNEL_SIZE,) * 2, np.uint8)
    return cv2.erode(cv2.dilate(cv2.bitwise_not(binary), kernel), kernel)


def find_board(binary: np.ndarray) -> BoardDetection:
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        approximation = cv2.approxPolyDP(
            contour, 0.02 * cv2.arcLength(contour, True), True
        )
        if len(approximation) == 4:
            points = approximation.reshape(4, 2).astype(np.float32)
            center = points.mean(axis=0)
            angles = np.arctan2(
                points[:, 1] - center[1], points[:, 0] - center[0]
            )
            ordered = points[np.argsort(angles)]
            top_left = np.argmin(ordered.sum(axis=1))
            ordered = np.roll(ordered, -top_left, axis=0)
            return BoardDetection(ordered, contour, cv2.contourArea(contour))
    raise ValueError("could not find a four-sided Sudoku board contour")


def rectify(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    destination = np.array(
        [[0, 0], [BOARD_SIZE - 1, 0], [BOARD_SIZE - 1, BOARD_SIZE - 1],
         [0, BOARD_SIZE - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.warpPerspective(grayscale, matrix, (BOARD_SIZE, BOARD_SIZE))


def extract_cell(board: np.ndarray, row: int, col: int) -> np.ndarray:
    crop_size = int(CELL_SIZE * (1 + CELL_MARGIN_RATIO))
    margin_before = (crop_size - CELL_SIZE) // 2
    margin_after = crop_size - CELL_SIZE - margin_before
    y1 = max(0, row * CELL_SIZE - margin_before)
    y2 = min(BOARD_SIZE, (row + 1) * CELL_SIZE + margin_after)
    x1 = max(0, col * CELL_SIZE - margin_before)
    x2 = min(BOARD_SIZE, (col + 1) * CELL_SIZE + margin_after)
    return board[y1:y2, x1:x2]


def preprocess_digit(
    cell: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, np.ndarray], float | None, int | None, str | None]:
    """Preprocess a cell and retain every image needed to diagnose failures."""
    threshold_source = cv2.GaussianBlur(cell, (5, 5), 0)
    threshold, _ = cv2.threshold(
        threshold_source, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    # Slightly raise the cutoff to retain more of the darker digit strokes.
    threshold = min(255, threshold + 5)
    _, binary = cv2.threshold(cell, threshold, 255, cv2.THRESH_BINARY_INV)
    cleared = clear_border(binary)
    contours, _ = cv2.findContours(cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.cvtColor(cleared, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        cv2.drawContours(contour_image, [contour], -1, (0, 180, 255), 2)
    debug_images = {
        "crop": cell.copy(),
        "processed": binary,
        "contours": contour_image,
    }
    if not contours:
        return None, debug_images, None, None, "no digit contour found"
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 3)
    x, y, width, height = cv2.boundingRect(contour)
    bounding_box_area = width * height
    if bounding_box_area < MIN_CONTOUR_AREA:
        return (
            None,
            debug_images,
            contour_area,
            bounding_box_area,
            "digit contour bounding-box area below threshold",
        )
    digit = cleared[y:y + height, x:x + width]
    scale = DIGIT_SIZE / max(width, height)
    resized = cv2.resize(
        digit, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    x_offset = (CANVAS_SIZE - resized.shape[1]) // 2
    y_offset = (CANVAS_SIZE - resized.shape[0]) // 2
    canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    debug_images["canvas"] = canvas
    return canvas, debug_images, contour_area, bounding_box_area, None


def load_model(path: Path):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def predict(model, digit: np.ndarray) -> tuple[int, float]:
    input_details = model.get_input_details()[0]
    output_details = model.get_output_details()[0]
    model.set_tensor(input_details["index"], digit.reshape(1, 28, 28, 1).astype(np.float32) / 255)
    model.invoke()
    probabilities = model.get_tensor(output_details["index"])[0]
    index = int(np.argmax(probabilities))
    return index + 1, float(probabilities[index])


def evaluate(path: Path, model, min_confidence: float) -> ImageResult:
    result = ImageResult(path)
    image = cv2.imread(str(path))
    if image is None:
        result.error = "could not load image"
        return result
    try:
        undistorted = undistort_image(image)
        detection = find_board(preprocess_board(undistorted))
        board = rectify(undistorted, detection.corners)
        contour_image = undistorted.copy()
        cv2.drawContours(contour_image, [detection.contour], -1, (0, 255, 0), 4)
        result.board_contour_area = detection.contour_area
        result.board_contour_image = contour_image
        result.rectified_image = board
        baseline = expected_grid(path)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                expected = int(baseline[row, col])
                (
                    digit,
                    debug_images,
                    contour_area,
                    contour_bounding_box_area,
                    preprocessing_reason,
                ) = preprocess_digit(extract_cell(board, row, col))
                cell = CellResult(
                    row,
                    col,
                    expected,
                    digit is not None,
                    contour_area=contour_area,
                    contour_bounding_box_area=contour_bounding_box_area,
                    debug_images=debug_images,
                )
                if digit is None:
                    if expected:
                        cell.reason = preprocessing_reason or "missed digit"
                        if contour_bounding_box_area is not None:
                            cell.reason += (
                                f" (threshold={MIN_CONTOUR_AREA}, "
                                f"measured={contour_bounding_box_area})"
                            )
                else:
                    predicted, cell.confidence = predict(model, digit)
                    if cell.confidence < min_confidence:
                        cell.predicted = None
                        if expected:
                            cell.reason = (
                                "OCR confidence below threshold "
                                f"(threshold={min_confidence:.6f}, "
                                f"measured={cell.confidence:.6f})"
                            )
                    else:
                        cell.predicted = predicted
                    if cell.predicted is None:
                        pass
                    elif not expected:
                        cell.reason = "false positive"
                    elif cell.predicted != expected:
                        cell.reason = "incorrect OCR"
                result.cells.append(cell)
    except (ValueError, RuntimeError, cv2.error) as exc:
        result.error = str(exc)
    return result


def save_debug_images(result: ImageResult, debug_dir: Path) -> int:
    """Save diagnostic stages for failed cells and return the number saved."""
    image_dir = debug_dir / result.path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    if result.board_contour_image is not None:
        board_output = image_dir / "board_contour.png"
        if not cv2.imwrite(str(board_output), result.board_contour_image):
            raise OSError(f"could not write debug image: {board_output}")
    if result.rectified_image is not None:
        rectified_output = image_dir / "rectified.png"
        if not cv2.imwrite(str(rectified_output), result.rectified_image):
            raise OSError(f"could not write debug image: {rectified_output}")
    saved = 0
    for cell in result.failures:
        cell_dir = image_dir / f"r{cell.row + 1}c{cell.col + 1}_{cell.reason.replace(' ', '_')}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        for name, image in cell.debug_images.items():
            output = image
            if name == "canvas":
                output = cv2.resize(image, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
            output_path = cell_dir / f"{name}.png"
            if not cv2.imwrite(str(output_path), output):
                raise OSError(f"could not write debug image: {output_path}")
        (cell_dir / "details.txt").write_text(
            f"cell: r{cell.row + 1}c{cell.col + 1}\n"
            f"expected: {cell.expected}\n"
            f"predicted: {cell.predicted}\n"
            f"confidence: {cell.confidence:.6f}\n"
            f"contour_area: {cell.contour_area}\n"
            f"contour_bounding_box_area: {cell.contour_bounding_box_area}\n"
            f"contour_area_threshold: {MIN_CONTOUR_AREA}\n"
            f"reason: {cell.reason}\n",
            encoding="utf-8",
        )
        saved += 1
    return saved


def show_debug_images(result: ImageResult) -> None:
    """Display diagnostic stages for failed cells until a key is pressed."""
    for cell in result.failures:
        label = f"{result.path.name} r{cell.row + 1}c{cell.col + 1}: {cell.reason}"
        for name, image in cell.debug_images.items():
            display = image
            if name == "canvas":
                display = cv2.resize(image, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"{label} - {name}", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def print_result(
    result: ImageResult,
    min_confidence: float,
    debug_dir: Path | None = None,
) -> None:
    print(f"\n{result.path.name}")
    if result.error:
        print(f"  ERROR: {result.error}")
        return
    if result.board_contour_area is not None:
        print(f"  original contour area: {result.board_contour_area:.2f}")
    if debug_dir is not None:
        print(f"  rectified image: {debug_dir / result.path.stem / 'rectified.png'}")
    counts = {}
    for cell in result.failures:
        counts[cell.reason] = counts.get(cell.reason, 0) + 1
    print(f"  failures: {len(result.failures)} ({', '.join(f'{key}: {value}' for key, value in counts.items()) or 'none'})")
    for cell in result.cells:
        if not cell.extracted and not cell.reason:
            continue
        location = f"r{cell.row + 1}c{cell.col + 1}"
        observed = "-" if cell.predicted is None else str(cell.predicted)
        confidence = f", confidence={cell.confidence:.3f}" if cell.extracted else ""
        correct = (
            (
                cell.expected > 0
                and cell.predicted == cell.expected
                and cell.confidence >= min_confidence
            )
            or (
                cell.expected == 0
                and cell.predicted is None
                and cell.extracted
                and cell.confidence < min_confidence
            )
        )
        reason = cell.reason or "correct"
        color = GREEN if correct else YELLOW
        print(
            f"{color}    {location}: expected={cell.expected}, "
            f"observed={observed}, {reason}{confidence}{RESET}"
        )
    if debug_dir is not None:
        saved = save_debug_images(result, debug_dir)
        print(
            f"  debug images: {debug_dir / result.path.stem} "
            f"(board contour + {saved} failed cells)"
        )


def print_summary(results: list[ImageResult], min_confidence: float) -> None:
    """Print aggregate OCR and preprocessing metrics for all image fixtures."""
    processed = [result for result in results if result.error is None]
    cells = [cell for result in processed for cell in result.cells]
    if not cells:
        return

    expected_digits = [cell for cell in cells if cell.expected > 0]
    empty_cells = [cell for cell in cells if cell.expected == 0]
    extracted = [cell for cell in cells if cell.extracted]
    accepted = [cell for cell in extracted if cell.predicted is not None]
    correct_digits = [
        cell for cell in expected_digits
        if cell.predicted == cell.expected and cell.confidence >= min_confidence
    ]
    false_positives = [cell for cell in empty_cells if cell.predicted is not None]
    extracted_empty = [cell for cell in empty_cells if cell.extracted]
    missed_digits = [cell for cell in expected_digits if not cell.extracted]
    rejected_digits = [
        cell for cell in expected_digits
        if cell.extracted and cell.predicted is None
    ]
    incorrect_digits = [
        cell for cell in expected_digits
        if cell.predicted is not None and cell.predicted != cell.expected
    ]

    failure_counts: dict[str, int] = {}
    for cell in cells:
        if cell.reason:
            failure_counts[cell.reason] = failure_counts.get(cell.reason, 0) + 1

    def percentage(part: int, whole: int) -> str:
        return f"{100 * part / whole:.1f}%" if whole else "n/a"

    correct_confidences = [cell.confidence for cell in correct_digits]
    false_positive_confidences = [cell.confidence for cell in extracted_empty]
    if extracted_empty:
        failure_counts["false positive candidate"] = len(extracted_empty)
    print("\nAggregate summary")
    print(f"  images: {len(processed)} processed, {len(results) - len(processed)} errors")
    print(f"  cells: {len(cells)} labelled ({len(expected_digits)} digits, {len(empty_cells)} empty)")
    print(
        f"  preprocessing: {len(extracted)}/{len(cells)} cells extracted "
        f"({percentage(len(extracted), len(cells))}), "
        f"{len(missed_digits)} expected digits missed"
    )
    print(
        f"  OCR candidates in empty cells: {len(extracted_empty)}/{len(empty_cells)} "
        f"({percentage(len(extracted_empty), len(empty_cells))})"
    )
    print(
        f"  accepted false positives: {len(false_positives)} "
        f"({percentage(len(false_positives), len(accepted))} of accepted predictions; "
        f"{percentage(len(false_positives), len(empty_cells))} of empty cells)"
    )
    print(
        f"  digit OCR: {len(correct_digits)}/{len(expected_digits)} correct "
        f"({percentage(len(correct_digits), len(expected_digits))}), "
        f"{len(rejected_digits)} rejected, {len(incorrect_digits)} incorrect"
    )
    if correct_confidences:
        print(f"  correct-digit confidence: min={min(correct_confidences):.3f}")
    if false_positive_confidences:
        print(f"  false-positive confidence: max={max(false_positive_confidences):.3f}")
    print(
        "  failure types: "
        + ", ".join(f"{reason}={count}" for reason, count in failure_counts.items())
        if failure_counts
        else "  failure types: none"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_PREDICTION_CONFIDENCE,
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=ROOT / "vision_debug",
        help="directory for failed-cell diagnostic images (default: ./vision_debug)",
    )
    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="display each failed cell's diagnostic images in OpenCV windows",
    )
    parser.add_argument("images", nargs="*", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if delete_debug:
        if args.debug_dir.exists() and not args.debug_dir.is_dir():
            print(f"Debug output path is not a directory: {args.debug_dir}", file=sys.stderr)
            return 2
        if args.debug_dir.is_dir():
            shutil.rmtree(args.debug_dir)
            print(f"Deleted debug output: {args.debug_dir}")
    paths = args.images or [
        args.image_dir / name
        for name in SELECTED_IMAGES
        if name in BASELINES and (args.image_dir / name).is_file()
    ]
    if not paths:
        print(f"No labelled images found in {args.image_dir}", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 2
    model = load_model(args.model)
    results = [evaluate(path, model, args.min_confidence) for path in paths]
    print(f"Vision evaluation ({len(results)} image(s), confidence >= {args.min_confidence:.2f})")
    for result in results:
        print_result(result, args.min_confidence, args.debug_dir)
        if args.show_debug and result.failures:
            show_debug_images(result)
    errors = [result for result in results if result.error]
    print_summary(results, args.min_confidence)
    print(f"\nSummary: {len(results) - len(errors)} processed, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())