# Sudoku-Solving Robot

A Python-based autonomous robot that uses computer vision to recognise and solve Sudoku puzzles, then physically writes the solutions onto the board.

## Project Structure

| File | Description |
|---|---|
| `main.py` | Main application and pipeline orchestration |
| `config.py` | Configuration and hardware parameters |
| `vision.py` | Image processing, board detection and digit recognition |
| `sudoku.py` | Sudoku representation, solving and printing |
| `geometry.py` | Coordinate transformations and pose estimation |
| `robot.py` | Differential drive and navigation |
| `pen.py` | Servo control and digit drawing |

## Pipeline

1. Capture an image using the Raspberry Pi camera.
2. Detect, rectify and extract the Sudoku board.
3. Recognise digits using a TensorFlow Lite CNN.
4. Construct and solve the Sudoku grid.
5. Estimate the robot's position relative to the board.
6. Generate a writing path through the empty cells.
7. Navigate and physically write the solution.

## Hardware

- Raspberry Pi Zero 2 W
- Camera module
- Two stepper motors (differential drive)
- Two servo motors (pen positioning and lifting)

## Dependencies

- Python
- OpenCV
- NumPy
- TensorFlow Lite
- Raspberry Pi hardware/PWM interfaces
