import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    own_score = current_to_customer + customer_to_depot

    other_trucks = np.array([p for idx, p in enumerate(truck_positions) if not np.allclose(p, current_position)])
    if len(other_trucks) == 0:
        best_idx = np.argmin(own_score)
        return int(best_idx)

    dist_to_other = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)
    nearest_other = np.min(dist_to_other, axis=1)

    beta = 1.5
    score = own_score - beta * nearest_other

    best_idx = np.argmin(score)
    return int(best_idx)