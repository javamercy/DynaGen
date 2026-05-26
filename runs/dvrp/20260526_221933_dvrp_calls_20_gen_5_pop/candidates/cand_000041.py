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
    # Distance from current truck to each customer
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    # Distance from each customer to depot
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # Identify index of current truck by closest position
    truck_dists = np.linalg.norm(truck_positions - current_position, axis=1)
    idx = np.argmin(truck_dists)
    # Compute distances from each customer to all trucks
    diff = available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)  # (n_available, n_trucks)
    # Exclude current truck
    dist_matrix[:, idx] = np.inf
    nearest_other = np.min(dist_matrix, axis=1)
    # Composite score with reduced depot weight
    depot_weight = 0.8
    score = dist_current - nearest_other + depot_weight * dist_depot
    min_score = np.min(score)
    # If the best score is positive, wait (return None)
    if min_score > 0:
        return None
    best_idx = np.argmin(score)
    return int(best_idx)