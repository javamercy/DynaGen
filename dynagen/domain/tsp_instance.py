from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from dynagen.domain.tour import is_valid_tour, tour_length, validate_tour


@dataclass
class TSPInstance:
    name: str
    dimension: int
    distance_matrix: np.ndarray
    optimal_length: float
    coordinates: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dimension = int(self.dimension)
        matrix = np.asarray(self.distance_matrix, dtype=float)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("Distance matrix shape must match dimension")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Distance matrix must contain finite values")
        if np.any(matrix < 0):
            raise ValueError("Distance matrix must be non-negative")
        if not np.allclose(np.diag(matrix), 0.0):
            raise ValueError("Distance matrix diagonal must be zero")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("v0 only supports symmetric TSP instances")

        self.distance_matrix = matrix
        if self.coordinates is not None:
            coordinates_arr = np.asarray(self.coordinates, dtype=float)
            if coordinates_arr.shape[0] != self.dimension or coordinates_arr.ndim != 2 or coordinates_arr.shape[1] < 2:
                raise ValueError("Coordinates must have shape (dimension, at least 2)")
            self.coordinates = coordinates_arr[:, :2]

        if self.optimal_length is None:
            raise ValueError("Optimal length must be provided")
        if not np.isfinite(self.optimal_length) or self.optimal_length < 0:
            raise ValueError("Optimal length must be a non-negative finite value")

    @classmethod
    def from_coordinates(
            cls,
            name: str,
            coordinates: Sequence[Sequence[float]] | np.ndarray,
            *,
            edge_weight_type: Literal["EUC_2D", "CEIL_2D"],
            optimal_length: float,
            metadata: dict[str, Any] | None = None,
    ) -> "TSPInstance":
        coordinates_arr = np.asarray(coordinates, dtype=float)
        matrix = euclidean_distance_matrix(coordinates_arr, edge_weight_type=edge_weight_type)
        meta = dict(metadata or {})
        meta.setdefault("edge_weight_type", edge_weight_type)
        return cls(
            name=name,
            dimension=int(coordinates_arr.shape[0]),
            coordinates=coordinates_arr[:, :2],
            distance_matrix=matrix,
            optimal_length=optimal_length,
            metadata=meta,
        )

    @classmethod
    def from_distance_matrix(
            cls,
            name: str,
            distance_matrix: Sequence[Sequence[float]] | np.ndarray,
            *,
            optimal_length: float,
            metadata: dict[str, Any] | None = None,
    ) -> "TSPInstance":
        matrix = np.asarray(distance_matrix, dtype=float)
        return cls(
            name=name,
            dimension=int(matrix.shape[0]),
            distance_matrix=matrix,
            optimal_length=optimal_length,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TSPInstance":
        return cls(
            name=data["name"],
            dimension=data["dimension"],
            coordinates=data.get("coordinates"),
            distance_matrix=data["distance_matrix"],
            optimal_length=data.get("optimal_length", float("inf")),
            metadata=dict(data.get("metadata", {})),
        )

    def validate_tour(self, tour: Sequence[int] | np.ndarray) -> np.ndarray:
        return validate_tour(tour, self.dimension)

    def is_valid_tour(self, tour: Sequence[int] | np.ndarray) -> bool:
        return is_valid_tour(tour, self.dimension)

    def tour_length(self, tour: Sequence[int] | np.ndarray) -> float:
        return tour_length(self.distance_matrix, tour)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "coordinates": None if self.coordinates is None else self.coordinates.tolist(),
            "distance_matrix": self.distance_matrix.tolist(),
            "optimal_length": self.optimal_length,
            "metadata": self.metadata,
        }


def euclidean_distance_matrix(
        coordinates: np.ndarray,
        *,
        edge_weight_type: Literal["EUC_2D", "CEIL_2D"]) -> np.ndarray:
    coordinates_arr = np.asarray(coordinates, dtype=float)

    if coordinates_arr.ndim != 2 or coordinates_arr.shape[1] != 2:
        raise ValueError("Coordinates must have shape (n, 2)")

    diff = coordinates_arr[:, np.newaxis, :] - coordinates_arr[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))

    if edge_weight_type == "EUC_2D":
        distances = np.floor(distances + 0.5)
    elif edge_weight_type == "CEIL_2D":
        distances = np.ceil(distances)
    else:
        raise ValueError(f"Unsupported coordinate edge weight type: {edge_weight_type}")

    np.fill_diagonal(distances, 0.0)
    return distances.astype(float)
