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

    # Identify other trucks
    mask = np.all(truck_positions == current_position, axis=1)
    other_trucks = truck_positions[~mask]

    if len(other_trucks) == 0:
        best = np.argmin(dist_current)
        return int(best)

    # Compute nearest other truck distance for each customer
    diff = available_customers[:, np.newaxis, :] - other_trucks[np.newaxis, :, :]
    dist_other = np.linalg.norm(diff, axis=2)
    nearest_other = np.min(dist_other, axis=1)

    # Dynamic beta based on current truck's depot distance relative to others
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    median_other = np.median(other_depot_dists)
    beta = 0.3 if current_depot_dist <= median_other else 0.7

    # Waiting condition: for all customers, dist_current > 1.15 * nearest_other
    # and current truck not too close to depot compared to others
    wait_condition = np.all(dist_current > 1.15 * nearest_other)
    max_other_depot = np.max(other_depot_dists)
    if wait_condition and current_depot_dist > 0.5 * max_other_depot:
        return None

    # Compute scores and pick best
    score = dist_current - nearest_other + beta * dist_depot
    best_idx = np.argmin(score)
    return int(best_idx)