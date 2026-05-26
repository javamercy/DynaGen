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

    # distances from each truck to depot
    all_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    # identify current truck index
    diff = truck_positions - current_position
    idx = np.where(np.all(np.isclose(diff, 0), axis=1))[0]
    if len(idx) == 0:
        # fallback (should not happen)
        idx = [np.argmin(np.linalg.norm(diff, axis=1))]
    current_idx = idx[0]

    current_ttt = np.max(all_depot_dists)

    best_customer = None
    best_ttt = float('inf')
    best_sum = float('inf')
    best_dist = float('inf')

    for i, cust in enumerate(available_customers):
        new_route = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # new array of depot distances
        new_dists = all_depot_dists.copy()
        new_dists[current_idx] = new_route
        candidate_ttt = np.max(new_dists)
        candidate_sum = np.sum(new_dists)
        dist_to_current = np.linalg.norm(current_position - cust)

        # Evaluate: lower ttt better; if tie, lower sum better; if tie, closer customer better
        if (candidate_ttt < best_ttt) or \
           (np.isclose(candidate_ttt, best_ttt) and candidate_sum < best_sum) or \
           (np.isclose(candidate_ttt, best_ttt) and np.isclose(candidate_sum, best_sum) and dist_to_current < best_dist):
            best_ttt = candidate_ttt
            best_sum = candidate_sum
            best_dist = dist_to_current
            best_customer = i

    # Wait if no customer improves the max return time (with small tolerance)
    if best_customer is None or best_ttt >= current_ttt - 1e-6:
        return None

    return best_customer