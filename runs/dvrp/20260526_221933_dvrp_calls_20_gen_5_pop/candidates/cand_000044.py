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
    # distances to customers
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # other trucks
    mask = ~np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[mask]
    # nearest other truck distance per customer
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
    else:
        diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)
        nearest_other = np.min(dist_other, axis=1)
    # waiting condition: if current truck's nearest customer is much farther than others'
    if len(other_trucks) > 0:
        # min distance per other truck to any customer
        other_min_per_truck = np.min(np.linalg.norm(available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :], axis=0), axis=0)
        min_other = np.min(other_min_per_truck)
        current_min = np.min(dist_current)
        if current_min > 1.5 * min_other:
            return None
    # time-dependent depot coefficient
    max_depot_dist = np.max(dist_depot) if len(dist_depot) > 0 else 1.0
    time_factor = min(1.0, current_time / (2 * max_depot_dist + 1e-6))
    depot_coeff = 0.3 * (1 + time_factor)
    # score
    score = dist_current - nearest_other + depot_coeff * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)