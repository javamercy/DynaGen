import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    if n_trucks == 1:
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    # Identify current truck index
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    # Distances from current position to customers
    current_dists = np.linalg.norm(available_customers - current_position, axis=1)
    # For each customer, distances to all other trucks
    all_dists = np.linalg.norm(available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2)
    # Set current truck's distance to inf for min computation
    all_dists[:, current_truck_idx] = np.inf
    other_min_dists = np.min(all_dists, axis=1)
    # Regret
    regrets = other_min_dists - current_dists
    # Depot distances
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    # Score: regret minus depot distance (prefer high regret, small depot distance)
    scores = regrets - depot_dists
    # Only consider customers where current truck is strictly closest (regret > 0)
    eligible = regrets > 0
    if np.any(eligible):
        best_idx = np.argmax(scores * eligible)  # eligible False gives 0, but scores may be negative so use mask
        # Actual max among eligible
        eligible_scores = scores[eligible]
        if eligible_scores.size > 0:
            best_local_idx = np.argmax(eligible_scores)
            best_global_idx = np.where(eligible)[0][best_local_idx]
            return int(best_global_idx)
    return None