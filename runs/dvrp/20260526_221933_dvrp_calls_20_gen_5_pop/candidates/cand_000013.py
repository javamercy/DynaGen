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
    n_trucks = truck_positions.shape[0]
    if n_trucks == 1:
        # No other trucks, fall back to simple score
        d_current = np.linalg.norm(available_customers - current_position, axis=1)
        d_depot = np.linalg.norm(available_customers - depot_position, axis=1)
        score = d_current + d_depot - 1e-6 * d_depot  # tie-break far customers
        best_idx = int(np.argmin(score))
        return best_idx

    # Distances from current truck
    d_current = np.linalg.norm(available_customers - current_position, axis=1)
    # Distances to depot
    d_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # For each customer, find minimum distance from any other truck
    # We need to compute for each other truck, then take min
    n_avail = len(available_customers)
    min_other = np.full(n_avail, np.inf)
    for i, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            continue
        dists = np.linalg.norm(available_customers - pos, axis=1)
        min_other = np.minimum(min_other, dists)
    # Penalty if current truck is farther than nearest other truck
    beta = 0.5
    penalty = beta * np.maximum(0, d_current - min_other)
    score = d_current + d_depot + penalty - 1e-6 * d_depot  # tie-break far customers
    best_idx = int(np.argmin(score))
    return best_idx