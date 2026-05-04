from collections.abc import Sequence

import numpy as np


def validate_tour(tour: Sequence[int] | np.ndarray, dimension: int) -> np.ndarray:
    arr = np.asarray(tour)
    if arr.ndim != 1:
        raise ValueError("Tour must be a one-dimensional sequence")
    if arr.size != dimension:
        raise ValueError(f"Tour length {arr.size} does not match dimension {dimension}")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("Tour entries must be integer node ids")
        if not np.all(np.isfinite(arr)) or not np.allclose(arr, np.round(arr)):
            raise ValueError("Tour entries must be integer node ids")

    nodes = arr.astype(int, copy=False)
    if np.any(nodes < 0) or np.any(nodes >= dimension):
        raise ValueError("Tour contains node ids outside the valid range")
    if len(set(int(node) for node in nodes)) != dimension:
        raise ValueError("Tour must be a permutation without repeated nodes")
    return nodes


def is_valid_tour(tour: Sequence[int] | np.ndarray, dimension: int) -> bool:
    try:
        validate_tour(tour, dimension)
    except ValueError:
        return False
    return True


def tour_length(distance_matrix: np.ndarray, tour: Sequence[int] | np.ndarray) -> float:
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Distance matrix must be square")

    nodes = validate_tour(tour, matrix.shape[0])
    next_nodes = np.roll(nodes, -1)
    return float(matrix[nodes, next_nodes].sum())
