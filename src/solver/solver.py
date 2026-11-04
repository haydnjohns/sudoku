#!/usr/bin/env python3
"""Dancing Links (Algorithm X) Sudoku solver, tuned for low-power hardware.

Public API
----------
    solve_sudoku(unsolved_sudoku, print_unsolved=False, print_solved=False) -> np.ndarray
    print_sudoku(sudoku, givens=None) -> None
    UnsolvableSudokuError            (subclass of ValueError)

Grid convention (input and output)
----------------------------------
    * numpy ndarray, shape (9, 9), any integer dtype
    * row-major: grid[row, col], row 0 is the top row, col 0 the left column
    * digits 1-9 are clues, 0 means "empty cell"
"""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np

__all__ = ["solve_sudoku", "print_sudoku", "UnsolvableSudokuError"]


class UnsolvableSudokuError(ValueError):
    """The puzzle is well formed but has no solution (e.g. clashing clues)."""


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------
def print_sudoku(sudoku: np.ndarray, givens: Optional[np.ndarray] = None) -> None:
    """Print a 9x9 grid to the terminal as ASCII art. Returns None.

    Parameters
    ----------
    sudoku : np.ndarray, shape (9, 9), integer values 0-9
        Grid to draw. 0 is drawn as '.'.
    givens : np.ndarray, shape (9, 9), optional
        The original puzzle. If supplied and stdout is a terminal, digits that
        are not in `givens` (i.e. the solver's additions) are drawn in green.
    """
    grid = _validate(sudoku)
    clue = _validate(givens) if givens is not None else None
    colour = clue is not None and sys.stdout.isatty()

    heavy = "#" + "#".join(["=" * 11] * 3) + "#"
    light = "#" + "#".join("+".join(["---"] * 3) for _ in range(3)) + "#"

    lines = [heavy]
    for r in range(9):
        cells = []
        for c in range(9):
            v = int(grid[r, c])
            ch = str(v) if v else "."
            if colour and v and not clue[r, c]:
                ch = f"\033[1;32m{ch}\033[0m"
            cells.append(f" {ch} ")
        lines.append("#" + "#".join("|".join(cells[b:b + 3]) for b in (0, 3, 6)) + "#")
        lines.append(heavy if r % 3 == 2 else light)
    print("\n".join(lines))


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def _validate(sudoku: np.ndarray) -> np.ndarray:
    arr = np.asarray(sudoku)
    if arr.shape != (9, 9):
        raise ValueError(f"sudoku must have shape (9, 9), got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"sudoku must have an integer dtype, got {arr.dtype}")
    if arr.min() < 0 or arr.max() > 9:
        raise ValueError("sudoku values must be in 0..9 (0 = empty)")
    return arr


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
def _solve_lists(grid: list) -> Optional[list]:
    """Dancing Links on a 9x9 list-of-lists. Returns solved list-of-lists or None.

    Optimisations: givens are folded into bitmasks so already-satisfied
    constraint columns are never created and clashing candidates never added
    (the matrix is far smaller than 729 x 324); links live in flat int lists;
    the search is iterative (no recursion overhead); columns are chosen by
    minimum remaining candidates.
    """
    row_used = [0] * 9
    col_used = [0] * 9
    box_used = [0] * 9
    empties = []
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v:
                bit = 1 << v
                b = (r // 3) * 3 + c // 3
                if (row_used[r] | col_used[c] | box_used[b]) & bit:
                    return None
                row_used[r] |= bit
                col_used[c] |= bit
                box_used[b] |= bit
            else:
                empties.append((r, c))

    # Constraint ids: 0-80 cell filled, 81-161 row/digit, 162-242 col/digit,
    # 243-323 box/digit. Only unsatisfied constraints become columns.
    active = [r * 9 + c for r, c in empties]
    for r in range(9):
        for d in range(1, 10):
            if not row_used[r] & (1 << d):
                active.append(81 + r * 9 + d - 1)
    for c in range(9):
        for d in range(1, 10):
            if not col_used[c] & (1 << d):
                active.append(162 + c * 9 + d - 1)
    for b in range(9):
        for d in range(1, 10):
            if not box_used[b] & (1 << d):
                active.append(243 + b * 9 + d - 1)
    active.sort()

    n = len(active)
    node_of = {cid: i + 1 for i, cid in enumerate(active)}  # node 0 = root

    # Link arrays: root = 0, column headers = 1..n, then row nodes.
    L = list(range(-1, n))
    R = list(range(1, n + 2))
    L[0] = n
    R[n] = 0
    U = list(range(n + 1))
    D = list(range(n + 1))
    C = list(range(n + 1))
    S = [0] * (n + 1)
    ROW = [None] * (n + 1)

    for r, c in empties:
        b = (r // 3) * 3 + c // 3
        blocked = row_used[r] | col_used[c] | box_used[b]
        for d in range(1, 10):
            if blocked & (1 << d):
                continue
            cols = (
                node_of[r * 9 + c],
                node_of[81 + r * 9 + d - 1],
                node_of[162 + c * 9 + d - 1],
                node_of[243 + b * 9 + d - 1],
            )
            first = len(L)
            for k, col in enumerate(cols):
                idx = first + k
                L.append(first + (k - 1) % 4)
                R.append(first + (k + 1) % 4)
                C.append(col)
                ROW.append((r, c, d))
                up = U[col]
                U.append(up)
                D.append(col)
                D[up] = idx
                U[col] = idx
                S[col] += 1

    def cover(c):
        l = L[c]
        r = R[c]
        R[l] = r
        L[r] = l
        i = D[c]
        while i != c:
            j = R[i]
            while j != i:
                u = U[j]
                d = D[j]
                D[u] = d
                U[d] = u
                S[C[j]] -= 1
                j = R[j]
            i = D[i]

    def uncover(c):
        i = U[c]
        while i != c:
            j = L[i]
            while j != i:
                S[C[j]] += 1
                U[D[j]] = j
                D[U[j]] = j
                j = L[j]
            i = U[i]
        L[R[c]] = c
        R[L[c]] = c

    sol = []
    while True:
        c = R[0]
        if c == 0:
            break  # every column covered -> solved

        best = S[c]
        j = R[c]
        while j != 0 and best > 1:
            if S[j] < best:
                best = S[j]
                c = j
            j = R[j]

        if best > 0:
            cover(c)
            r = D[c]
            sol.append(r)
            j = R[r]
            while j != r:
                cover(C[j])
                j = R[j]
            continue

        while True:  # dead end: backtrack
            if not sol:
                return None
            r = sol.pop()
            j = L[r]
            while j != r:
                uncover(C[j])
                j = L[j]
            c = C[r]
            r = D[r]
            if r != c:  # try the next candidate in this column
                sol.append(r)
                j = R[r]
                while j != r:
                    cover(C[j])
                    j = R[j]
                break
            uncover(c)  # column exhausted, keep unwinding

    out = [row[:] for row in grid]
    for node in sol:
        r, c, d = ROW[node]
        out[r][c] = d
    return out


def solve_sudoku(
    unsolved_sudoku: np.ndarray,
    print_unsolved: bool = False,
    print_solved: bool = False,
) -> np.ndarray:
    """Solve a Sudoku with Dancing Links.

    Parameters
    ----------
    unsolved_sudoku : np.ndarray, shape (9, 9), integer dtype
        Puzzle: digits 1-9 are clues, 0 is empty. Not modified.
    print_unsolved : bool, default False
        Print the puzzle before solving.
    print_solved : bool, default False
        Print the solution after solving (new digits highlighted on a terminal).

    Returns
    -------
    np.ndarray, shape (9, 9)
        The solved grid (all cells 1-9), same dtype as the input. A new array.
        If the puzzle has several solutions, one of them is returned.

    Raises
    ------
    ValueError
        Input is not a (9, 9) integer array with values in 0..9.
    UnsolvableSudokuError
        (subclass of ValueError) The puzzle has no solution.
    """
    puzzle = _validate(unsolved_sudoku)
    if print_unsolved:
        print_sudoku(puzzle)

    solved_list = _solve_lists(puzzle.tolist())  # plain lists: much faster than numpy scalars
    if solved_list is None:
        raise UnsolvableSudokuError("sudoku has no solution")
    solved = np.array(solved_list, dtype=puzzle.dtype)

    if print_solved:
        print_sudoku(solved, givens=puzzle)
    return solved


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    demo = np.array([
        [0, 6, 0, 0, 0, 0, 1, 0, 0],
        [0, 7, 9, 6, 0, 0, 0, 0, 5],
        [0, 0, 4, 0, 9, 5, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 8, 2, 0],
        [0, 8, 0, 0, 0, 1, 0, 0, 4],
        [0, 0, 0, 0, 7, 0, 0, 0, 0],
        [0, 3, 5, 0, 0, 0, 0, 0, 0],
        [0, 0, 7, 0, 6, 2, 0, 0, 9],
        [0, 0, 0, 0, 0, 9, 0, 4, 7],
    ])
    t0 = time.perf_counter()
    result = solve_sudoku(demo, print_unsolved=True, print_solved=True)
    print(f"Solved in {(time.perf_counter() - t0) * 1000:.3f} ms")
