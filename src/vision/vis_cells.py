"""Extract Sudoku cells and perform OCR."""

import sys
from pathlib import Path
import cv2
import numpy as np
from skimage.segmentation import clear_border
if __package__ in (None, ""): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

MODEL_PATH = config.ASSETS_DIR / "vis_model.tflite"
DEMO_IMAGE_PATH = config.DEMO_IMAGE_PATH
BOARD_IMAGE_SIZE = config.BOARD_IMAGE_SIZE #900
GRID_SIZE = config.GRID_SIZE
DIGIT_IMAGE_SIZE = 28
DIGIT_CANVAS_SIZE = 28
DIGIT_SIZE = 20
MIN_CONTOUR_AREA = 500
CELL_MARGIN_RATIO = 0.25
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11
ADAPTIVE_THRESHOLD_C = 2
DEBUG = 0


def extract_cell(board_image: np.ndarray, row: int, col: int) -> np.ndarray:
    """Extract one cell with a margin, clipped to the board image."""

    cell_size = BOARD_IMAGE_SIZE // GRID_SIZE
    crop_size = int(cell_size * (1 + CELL_MARGIN_RATIO))
    margin_before = (crop_size - cell_size) // 2
    margin_after = crop_size - cell_size - margin_before

    y1 = max(0, row * cell_size - margin_before)
    y2 = min(board_image.shape[0], (row + 1) * cell_size + margin_after)
    x1 = max(0, col * cell_size - margin_before)
    x2 = min(board_image.shape[1], (col + 1) * cell_size + margin_after)
    cell_image = board_image[y1:y2, x1:x2]

    return cell_image


def preprocess_digit(cell_image: np.ndarray) -> np.ndarray | None:

    """Return preprocessing images and the final OCR canvas, if detected."""

    binary = cv2.adaptiveThreshold(
        cell_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_THRESHOLD_BLOCK_SIZE,
        ADAPTIVE_THRESHOLD_C
    )

    # if DEBUG:
    #     cv2.imshow("Preprocessed Digit", binary)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    border_cleared = clear_border(binary)
    contours, _ = cv2.findContours(border_cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)




    digit = None
    resized_digit = None
    canvas = np.zeros((DIGIT_CANVAS_SIZE, DIGIT_CANVAS_SIZE), dtype=np.uint8)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        if width * height >= MIN_CONTOUR_AREA:
            digit = border_cleared[y:y + height, x:x + width]
            scale = DIGIT_SIZE / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_digit = cv2.resize(digit, (new_width, new_height), interpolation=cv2.INTER_AREA)
            x_offset = (DIGIT_CANVAS_SIZE - new_width) // 2
            y_offset = (DIGIT_CANVAS_SIZE - new_height) // 2
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_digit

            if DEBUG:
                cv2.imshow("Max contour check passed", canvas)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return canvas if digit is not None else None


def load_digit_model():

    """Load the TFLite digit recognition model."""

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    return interpreter


def predict_digit(digit_image: np.ndarray, model) -> int:

    """Predict a digit using the TFLite model."""

    input_data = digit_image.reshape(1, DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE, 1)
    input_data = input_data.astype(np.float32) / 255.0

    input_details = model.get_input_details()[0]
    output_details = model.get_output_details()[0]
    model.set_tensor(input_details["index"], input_data)
    model.invoke()
    prediction = model.get_tensor(output_details["index"])

    predicted_digit = int(np.argmax(prediction[0])) + 1

    return predicted_digit


def recognise_board(board_image: np.ndarray) -> np.ndarray:

    """Create a 9x9 NumPy array representing the Sudoku board from rectified board image."""

    model = load_digit_model()
    board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell = extract_cell(board_image, row, col)
            digit_image = preprocess_digit(cell)
            if digit_image is not None:
                board[row, col] = predict_digit(digit_image, model)

    return board
