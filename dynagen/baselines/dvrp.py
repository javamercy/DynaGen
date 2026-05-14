from __future__ import annotations

import numpy as np


DVRP_BASELINES = {
    "greedy": r'''import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers, current_time, seed, budget):
    if len(available_customers) == 0:
        return None

    res = np.argmin(np.linalg.norm(available_customers - current_position, axis=1))
    return res
''',
    "heuristic": r'''import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers, current_time, seed, budget):
    scores = []
    for customer_pos in available_customers:
        distance = np.linalg.norm(current_position - customer_pos)
        vector_to_customer = customer_pos - current_position
        vector_to_depot = depot_position - current_position
        dot_product = np.dot(vector_to_customer, vector_to_depot)
        score = distance + 2 * dot_product
        scores.append(score)

    return int(np.argmin(scores))
''',
}


def get_dvrp_baseline_code(name: str) -> str:
    try:
        return DVRP_BASELINES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown DVRP baseline: {name}") from exc
