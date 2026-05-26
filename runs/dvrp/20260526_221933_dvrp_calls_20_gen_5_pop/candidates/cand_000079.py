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
    other_trucks = truck_positions[~mask] if mask.any() else truck_positions

    if len(other_trucks) == 0:
        score = dist_current + 0.3 * dist_depot
        return int(np.argmin(score))

    # Nearest other truck distance for each customer
    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)
    nearest_other = np.min(dist_other, axis=1)

    # Dynamic depot coefficient based on truck's depot distance
    truck_depot_dist = np.linalg.norm(current_position - depot_position)
    all_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    mean_depot = np.mean(all_depot_dists)
    if mean_depot < 1e-6:
        depot_coefficient = 0.3
    else:
        depot_coefficient = 0.3 + 0.2 * (truck_depot_dist / mean_depot)
        depot_coefficient = min(depot_coefficient, 0.8)

    # Adjust for remaining customers
    customers_per_truck = len(available_customers) / len(truck_positions)
    if customers_per_truck < 2:
        depot_coefficient = min(depot_coefficient + 0.1, 1.0)

    # Waiting condition: compare current truck's nearest customer to other trucks' nearest customers
    current_min_dist = np.min(dist_current)
    other_min_dists = [np.min(np.linalg.norm(available_customers - truck, axis=1)) for truck in other_trucks]
    mean_other_min = np.mean(other_min_dists) if other_min_dists else 0.0

    wait_threshold = 1.5 if customers_per_truck >= 2 else 2.0
    if mean_other_min > 0 and current_min_dist > wait_threshold * mean_other_min:
        return None

    score = dist_current - nearest_other + depot_coefficient * dist_depot
    return int(np.argmin(score))