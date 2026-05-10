import random

import numpy as np

from dynagen.domain.tsp_instance import TSPInstance


def generate_llamea_tsp_instance(*, seed: int = 69, size: int = 32) -> TSPInstance:
    """Build the synthetic TSP instance used by LLaMEA's TSP example."""
    rng = random.Random(seed)
    coordinates = [(50.0, 50.0)]
    for _ in range(size):
        coordinates.append((float(rng.randint(0, 100)), float(rng.randint(0, 100))))
        rng.randint(10, 35)

    coordinates_arr = np.asarray(coordinates, dtype=float)
    diff = coordinates_arr[:, np.newaxis, :] - coordinates_arr[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(distances, 0.0)

    return TSPInstance(
        name=f"llamea_seed{seed}_size{size}",
        dimension=size + 1,
        coordinates=coordinates_arr,
        distance_matrix=distances,
        optimal_length=None,
        metadata={
            "source": f"synthetic:llamea:{seed}:{size}",
            "generator": "generate_tsp_test",
            "seed": int(seed),
            "customer_count": int(size),
            "depot": [50.0, 50.0],
        },
    )


def parse_llamea_tsp_spec(spec: str) -> tuple[int, int] | None:
    parts = spec.split(":")
    if len(parts) != 4 or parts[:2] != ["synthetic", "llamea"]:
        return None
    return int(parts[2]), int(parts[3])
