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
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    if len(other_trucks) == 0:
        best_idx = np.argmin(own_score)
        return int(best_idx)
    else:
        # Waiting condition
        min_own = np.min(own_score)
        other_to_depot = np.linalg.norm(other_trucks - depot_position, axis=1)
        max_other_depot = np.max(other_to_depot)
        if min_own < 0.5 * max_other_depot:
            return None
        # Original score with isolation
        dist_to_other = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)
        nearest_other = np.min(dist_to_other, axis=1)
        beta = 0.5
        score = own_score - beta * nearest_other
        best_idx = np.argmin(score)
        return int(best_idx)