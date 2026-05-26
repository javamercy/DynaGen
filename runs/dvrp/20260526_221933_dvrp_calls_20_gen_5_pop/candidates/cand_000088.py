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

    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    mask = np.all(truck_positions == current_position, axis=1)
    if mask.any():
        other_trucks = truck_positions[~mask]
    else:
        other_trucks = truck_positions

    if len(other_trucks) == 0:
        # Only one truck: balance distance and depot
        score = dist_current + 0.3 * dist_depot
        best_idx = np.argmin(score)
        return int(best_idx)

    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)
    nearest_other = np.min(dist_other, axis=1)

    max_dist_to_depot = np.max(dist_depot)
    if max_dist_to_depot < 1e-6:
        depot_coefficient = 0.3
    else:
        # Logistic function: range [0.3, 0.7], midpoint at 0.5*max_dist_to_depot, steepness 0.1
        midpoint = 0.5 * max_dist_to_depot
        k = 0.1
        depot_coefficient = 0.3 + 0.4 / (1 + np.exp(-k * (current_time - midpoint)))
        depot_coefficient = min(depot_coefficient, 0.7)

    # Lower threshold to reduce waiting (more exploitation)
    wait_threshold = 1.2
    if np.all(dist_current > wait_threshold * nearest_other):
        return None

    score = dist_current - nearest_other + depot_coefficient * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)