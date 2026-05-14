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
    specs = parse_llamea_tsp_specs(spec)
    if specs is None:
        return None
    if len(specs) != 1:
        raise ValueError(
            "Expected one synthetic LLaMEA TSP instance. "
            "Use parse_llamea_tsp_specs for multi-instance specs."
        )
    return specs[0]


def parse_llamea_tsp_specs(spec: str) -> list[tuple[int, int]] | None:
    parts = spec.split(":")
    if len(parts) != 4 or parts[:2] != ["synthetic", "llamea"]:
        return None

    seeds = _parse_int_axis(parts[2], names=("seed", "seeds"))
    sizes = _parse_int_axis(parts[3], names=("size", "sizes"))
    return [(seed, size) for seed in seeds for size in sizes]


def _parse_int_axis(value: str, *, names: tuple[str, ...]) -> list[int]:
    value = value.strip()
    if "=" in value:
        key, raw_values = value.split("=", 1)
        if key.strip() not in names:
            expected = " or ".join(names)
            raise ValueError(f"Expected {expected}=... in synthetic LLaMEA TSP spec, got {key!r}")
        value = raw_values

    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        expected = " or ".join(names)
        raise ValueError(f"Expected at least one integer for {expected} in synthetic LLaMEA TSP spec")

    return [int(item) for item in values]
