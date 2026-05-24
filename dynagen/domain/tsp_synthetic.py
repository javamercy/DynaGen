import numpy as np

from dynagen.domain.tsp_instance import TSPInstance


def generate_tsp_construct_instances(
        *,
        n_instance: int = 20,
        n_cities: int = 500,
        seed: int = 2024,
) -> list[TSPInstance]:
    n_instance = int(n_instance)
    n_cities = int(n_cities)
    seed = int(seed)
    if n_instance < 1:
        raise ValueError("n_instance must be at least 1")
    if n_cities < 2:
        raise ValueError("n_cities must be at least 2")

    rng = np.random.RandomState(seed)
    instances: list[TSPInstance] = []
    source = f"synthetic:tsp_construct:n_instance={n_instance}:n_cities={n_cities}:seed={seed}"

    for index in range(n_instance):
        coordinates = rng.rand(n_cities, 2)
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(distances, 0.0)
        instances.append(TSPInstance(
            name=f"tsp_construct_{n_cities}_seed{seed}_{index:03d}",
            dimension=n_cities,
            coordinates=coordinates,
            distance_matrix=distances,
            optimal_length=None,
            metadata={
                "source": source,
                "generator": "generate_tsp_construct_instances",
                "seed": seed,
                "n_instance": n_instance,
                "n_cities": n_cities,
                "instance_index": index,
            },
        ))

    return instances


def parse_tsp_construct_spec(spec: str) -> tuple[int, int, int] | None:
    parts = spec.split(":")
    if len(parts) != 5 or parts[:2] != ["synthetic", "tsp_construct"]:
        return None

    n_instance = _parse_int_field(parts[2], names=("n_instance", "count", "instances"))
    n_cities = _parse_int_field(parts[3], names=("n_cities", "size", "problem_size"))
    seed = _parse_int_field(parts[4], names=("seed",))
    return n_instance, n_cities, seed


def _parse_int_field(value: str, *, names: tuple[str, ...]) -> int:
    value = value.strip()
    if "=" in value:
        key, raw_value = value.split("=", 1)
        if key.strip() not in names:
            expected = " or ".join(names)
            raise ValueError(f"Expected {expected}=... in synthetic tsp_construct spec, got {key!r}")
        value = raw_value

    value = value.strip()
    if not value:
        expected = " or ".join(names)
        raise ValueError(f"Expected a value for {expected} in synthetic tsp_construct spec")
    return int(value)
