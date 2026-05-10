from pathlib import Path
from typing import Any

import numpy as np

from dynagen.domain.tsp_instance import TSPInstance

_SUPPORTED_EDGE_WEIGHT_TYPES = {"EUC_2D", "CEIL_2D"}


def load_tsplib_file(path: str | Path) -> TSPInstance:
    path = Path(path)
    return parse_tsplib(path.read_text(encoding="utf-8"), source=str(path))


def parse_tsplib(text: str, *, source: str | None = None) -> TSPInstance:
    headers: dict[str, str] = {}
    coordinates: list[tuple[int, float, float]] = []
    weights: list[float] = []
    section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        upper = line.upper()

        if upper == "EOF":
            break

        if upper in {"NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION"}:
            section = upper
            continue

        if section == "NODE_COORD_SECTION":
            node_id, x, y = line.split()[:3]
            coordinates.append((int(node_id), float(x), float(y)))
            continue

        if section == "EDGE_WEIGHT_SECTION":
            weights.extend(float(value) for value in line.split())
            continue

        key, value = parse_header(line)
        headers[key.upper()] = value

    name = headers.get("NAME", "unnamed")
    dimension = int(require_header(headers, "DIMENSION"))
    tsp_type = headers.get("TYPE", "TSP").upper()
    edge_weight_type = headers.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    optimal_length = parse_optional_number(headers.get("OPTIMAL"))

    if tsp_type != "TSP":
        raise ValueError(f"Only symmetric TSP instances are supported, got TYPE={tsp_type}")

    metadata: dict[str, Any] = {
        **headers,
        "source": source,
    }

    if edge_weight_type in _SUPPORTED_EDGE_WEIGHT_TYPES:
        if len(coordinates) != dimension:
            raise ValueError("NODE_COORD_SECTION count does not match DIMENSION")

        coordinates_arr = coordinates_to_array(coordinates)

        return TSPInstance.from_coordinates(
            name=name,
            coordinates=coordinates_arr,
            edge_weight_type=edge_weight_type,  # type: ignore[arg-type]
            optimal_length=optimal_length,
            metadata=metadata,
        )

    if edge_weight_type == "EXPLICIT":
        matrix = full_matrix(weights, dimension)

        return TSPInstance.from_distance_matrix(
            name=name,
            distance_matrix=matrix,
            optimal_length=optimal_length,
            metadata=metadata,
        )

    raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")


def parse_header(line: str) -> tuple[str, str]:
    if ":" in line:
        key, value = line.split(":", 1)
        return key.strip(), value.strip()

    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Invalid TSPLIB header line: {line}")

    return parts[0].strip(), parts[1].strip()


def require_header(headers: dict[str, str], key: str) -> str:
    try:
        return headers[key]
    except KeyError:
        raise ValueError(f"TSPLIB file is missing {key}") from None


def parse_optional_number(value: str | None) -> float | None:
    if value is None:
        return None
    return float(value)


def coordinates_to_array(coordinates: list[tuple[int, float, float]]) -> np.ndarray:
    coordinates = sorted(coordinates, key=lambda item: item[0])
    return np.array([[x, y] for _, x, y in coordinates], dtype=float)


def full_matrix(weights: list[float], dimension: int) -> np.ndarray:
    expected = dimension * dimension

    if len(weights) != expected:
        raise ValueError(f"FULL_MATRIX expected {expected} weights, got {len(weights)}")

    return np.asarray(weights, dtype=float).reshape((dimension, dimension))
