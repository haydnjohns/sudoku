
import cv2
import numpy as np
from skimage.segmentation import clear_border

import config


# --------------------------------------------------------------------------
# Cell extraction
# --------------------------------------------------------------------------

def extract_cell(
    board_image: np.ndarray,
    row: int,
    col: int
) -> np.ndarray:
    """Extract one cell, excluding surrounding grid lines."""
    cell_size = config.BOARD_IMAGE_SIZE // config.GRID_SIZE

    buffer = int(cell_size * 0.1)

    y1 = row * cell_size + buffer
    y2 = (row + 1) * cell_size - buffer
    x1 = col * cell_size + buffer
    x2 = (col + 1) * cell_size - buffer

    return board_image[y1:y2, x1:x2]


# --------------------------------------------------------------------------
# Digit preprocessing
# --------------------------------------------------------------------------

def preprocess_digit(cell_image: np.ndarray) -> np.ndarray | None:
    """Extract and centre a digit on a 28x28 image."""
    grayscale = cv2.cvtColor(
        cell_image,
        cv2.COLOR_BGR2GRAY
    )

    binary = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    binary = clear_border(binary)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < config.MIN_DIGIT_AREA:
        return None

    x, y, w, h = cv2.boundingRect(contour)

    digit = binary[y:y + h, x:x + w]

    scale = config.DIGIT_SIZE / max(w, h)

    new_width = int(w * scale)
    new_height = int(h * scale)

    digit = cv2.resize(
        digit,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (config.DIGIT_CANVAS_SIZE, config.DIGIT_CANVAS_SIZE),
        dtype=np.uint8
    )

    x_offset = (config.DIGIT_CANVAS_SIZE - new_width) // 2
    y_offset = (config.DIGIT_CANVAS_SIZE - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = digit

    return canvas


# --------------------------------------------------------------------------
# Digit recognition
# --------------------------------------------------------------------------

def load_digit_model():
    """Load the TFLite digit recognition model."""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(
        model_path=str(config.MODEL_PATH)
    )

    interpreter.allocate_tensors()

    return interpreter


def predict_digit(
    digit_image: np.ndarray,
    interpreter
) -> int:
    """Predict a digit using the TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = digit_image.reshape(
        1,
        config.DIGIT_IMAGE_SIZE,
        config.DIGIT_IMAGE_SIZE,
        1
    )

    input_data = input_data.astype(np.float32) / 255.0

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )

    return int(np.argmax(prediction)) + 1


# --------------------------------------------------------------------------
# Board recognition
# --------------------------------------------------------------------------

def recognise_board(board_image: np.ndarray) -> np.ndarray:
    """
    Recognise digits in a rectified Sudoku board.

    Returns a 9x9 NumPy array.
    Empty cells are represented by zero.
    """
    interpreter = load_digit_model()

    board = np.zeros(
        (config.GRID_SIZE, config.GRID_SIZE),
        dtype=np.int32
    )

    for row in range(config.GRID_SIZE):
        for col in range(config.GRID_SIZE):
            cell = extract_cell(
                board_image,
                row,
                col
            )

            digit_image = preprocess_digit(cell)

            if digit_image is not None:
                board[row, col] = predict_digit(
                    digit_image,
                    interpreter
                )

    return board


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from vis_board import process_image

    image = cv2.imread(str(config.DEMO_IMAGE_PATH))

    rectified = process_image(image, debug=True)

    board = recognise_board(rectified.image)

    print("Recognised Sudoku:")
    print(board)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
