import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    n_cust = len(available_customers)
    n_trucks = len(truck_positions)

    # distances from current position to each customer
    dist_curr = np.linalg.norm(available_customers - current_position, axis=1)

    # find current truck index
    truck_idx = None
    for i, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            truck_idx = i
            break
    if truck_idx is None:
        # fallback: use closest truck
        dists = np.linalg.norm(truck_positions - current_position, axis=1)
        truck_idx = int(np.argmin(dists))

    # compute distances from all trucks to all customers
    all_dists = np.zeros((n_trucks, n_cust))
    for t in range(n_trucks):
        all_dists[t] = np.linalg.norm(available_customers - truck_positions[t], axis=1)

    # for each customer, compute minimum distance from trucks other than current
    other_min = np.full(n_cust, np.inf)
    for c in range(n_cust):
        for t in range(n_trucks):
            if t != truck_idx and all_dists[t, c] < other_min[c]:
                other_min[c] = all_dists[t, c]

    # regret = best alternative distance - distance from current truck
    regret = other_min - dist_curr

    # median distance from current position to customers
    median_dist = np.median(dist_curr)
    threshold = 0.5 * median_dist

    best_regret = np.max(regret)
    if best_regret > threshold:
        best_idx = int(np.argmax(regret))
        return best_idx
    else:
        return None