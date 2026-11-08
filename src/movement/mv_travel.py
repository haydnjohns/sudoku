"""Construct the board-relative path used to stamp solved Sudoku cells."""

from dataclasses import dataclass
from math import atan2, degrees, hypot
from time import sleep

import numpy as np

import config
from movement import mv_wheels
from vision.vis_board import BoardPose


@dataclass(frozen=True)
class StampTarget:
    """A board-relative position and the digit to stamp there.

    ``digit`` is ``None`` for the initial robot position.
    """

    x: float
    y: float
    digit: int | None = None
    row: int | None = None
    column: int | None = None
    heading_degrees: float = 0.0


@dataclass(frozen=True)
class TravelMove:
    """One turn-drive-turn movement in board-relative coordinates."""

    start_x: float
    start_y: float
    target_x: float
    target_y: float
    first_turn_degrees: float
    drive_distance_mm: float
    final_turn_degrees: float
    reverse: bool = False


def calculate_cell_dimensions(board_pose: BoardPose) -> tuple[float, float]:
    """Return cell width and height in millimetres."""
    if config.GRID_SIZE <= 0:
        raise ValueError("GRID_SIZE must be positive")
    return (
        (board_pose.dx_top + board_pose.dx_btm) / (2 * config.GRID_SIZE),
        (board_pose.dy_left + board_pose.dy_top) / (2 * config.GRID_SIZE),
    )


def _cell_center(
    row: int,
    column: int,
    cell_width: float,
    cell_height: float,
) -> tuple[float, float]:
    """Return a cell centre in board coordinates.

    Sudoku arrays are indexed from the visual top row, while board Y
    coordinates increase from the bottom edge.
    """
    return (
        (column + 0.5) * cell_width,
        (config.GRID_SIZE - row - 0.5) * cell_height,
    )


def build_stamp_path(
    board_pose: BoardPose,
    puzzle: np.ndarray,
    solution: np.ndarray,
) -> list[StampTarget]:
    """Build the bottom-to-top, left-to-right stamping path.

    ``puzzle`` contains the original grid, where zero denotes an empty cell.
    ``solution`` contains the solved grid. Coordinates are for the stamper;
    the wheel-centre setback is intentionally not applied here.
    """
    expected_shape = (config.GRID_SIZE, config.GRID_SIZE)
    if puzzle.shape != expected_shape or solution.shape != expected_shape:
        raise ValueError(
            f"puzzle and solution must have shape {expected_shape}"
        )
    if np.any(puzzle < 0) or np.any(solution < 0):
        raise ValueError("puzzle and solution must contain non-negative values")

    cell_width, cell_height = calculate_cell_dimensions(board_pose)
    path = [
        StampTarget(
            x=board_pose.x,
            y=board_pose.y,
            heading_degrees=board_pose.theta,
        )
    ]

    for row in range(config.GRID_SIZE - 1, -1, -1):
        for column in range(config.GRID_SIZE):
            if puzzle[row, column] != 0:
                continue
            x, y = _cell_center(row, column, cell_width, cell_height)
            path.append(
                StampTarget(
                    x=x,
                    y=y,
                    digit=int(solution[row, column]),
                    row=row,
                    column=column,
                )
            )

    return path


def _normalise_angle(angle: float) -> float:
    """Return an angle in the range [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def _wheel_centre_target(target: StampTarget) -> tuple[float, float]:
    """Convert a stamper target to a wheel-centre target.

    The signed setback is resolved along the target heading, so this also
    works for the initial target before the robot is aligned with +X.
    """
    heading = np.deg2rad(target.heading_degrees)
    return (
        target.x - config.STAMPER_SETBACK * np.cos(heading),
        target.y - config.STAMPER_SETBACK * np.sin(heading),
    )


def plan_travel_moves(
    path: list[StampTarget],
    *,
    debug: bool = False,
) -> list[TravelMove]:
    """Plan turn-drive-turn moves for stamper positions.

    The first path point supplies the current stamper position and heading.
    Every later target defaults to the required final heading of +X.
    """
    if len(path) < 2:
        return []

    current = path[0]
    current_x, current_y = _wheel_centre_target(current)
    current_heading = current.heading_degrees
    moves: list[TravelMove] = []

    for target in path[1:]:
        target_x, target_y = _wheel_centre_target(target)
        dx = target_x - current_x
        dy = target_y - current_y

        same_row = (
            abs(dy) < config.ROW_TRAVEL_TOLERANCE
            and abs(_normalise_angle(current_heading)) < 1e-6
        )
        if same_row:
            move = TravelMove(
                start_x=current_x,
                start_y=current_y,
                target_x=target_x,
                target_y=target_y,
                first_turn_degrees=0.0,
                drive_distance_mm=dx,
                final_turn_degrees=0.0,
            )
        else:
            distance = hypot(dx, dy)
            if distance == 0:
                raise ValueError("Cannot plan a move to the current position")

            if not moves:
                travel_heading = degrees(atan2(dy, dx))
                move = TravelMove(
                    start_x=current_x,
                    start_y=current_y,
                    target_x=target_x,
                    target_y=target_y,
                    first_turn_degrees=_normalise_angle(
                        travel_heading - current_heading
                    ),
                    drive_distance_mm=distance,
                    final_turn_degrees=_normalise_angle(
                        target.heading_degrees - travel_heading
                    ),
                )
            else:
                # Reverse along the vector toward the next row. The robot
                # points opposite to travel, then finishes facing +X.
                reverse_heading = degrees(atan2(-dy, -dx))
                move = TravelMove(
                    start_x=current_x,
                    start_y=current_y,
                    target_x=target_x,
                    target_y=target_y,
                    first_turn_degrees=_normalise_angle(
                        reverse_heading - current_heading
                    ),
                    drive_distance_mm=distance,
                    final_turn_degrees=_normalise_angle(
                        target.heading_degrees - reverse_heading
                    ),
                    reverse=True,
                )

        moves.append(move)
        if debug:
            print(
                f"({move.start_x:.1f}, {move.start_y:.1f}) -> "
                f"({move.target_x:.1f}, {move.target_y:.1f}): "
                f"turn {move.first_turn_degrees:.1f}°, "
                f"{'reverse ' if move.reverse else ''}"
                f"drive {move.drive_distance_mm:.1f} mm, "
                f"turn {move.final_turn_degrees:.1f}°"
            )
        current_x, current_y = target_x, target_y
        current_heading = target.heading_degrees

    return moves


def execute_travel_path(
    path: list[StampTarget],
    *,
    debug: bool = False,
) -> None:
    """Execute a planned path using centre pivots and straight wheel motion."""
    moves = plan_travel_moves(path, debug=debug)

    for move, target in zip(moves, path[1:]):
        if move.first_turn_degrees:
            mv_wheels.pivot("centre", move.first_turn_degrees)
            sleep(config.MOVEMENT_WAIT_SECONDS)

        distance_mm = move.drive_distance_mm
        if move.reverse:
            distance_mm = -distance_mm
        if distance_mm:
            mv_wheels.move_straight(distance_mm)
            sleep(config.MOVEMENT_WAIT_SECONDS)

        if move.final_turn_degrees:
            mv_wheels.pivot("centre", move.final_turn_degrees)
            sleep(config.MOVEMENT_WAIT_SECONDS)

        if target.digit is not None:
            # stampcell()
            sleep(config.STAMP_WAIT_SECONDS)
