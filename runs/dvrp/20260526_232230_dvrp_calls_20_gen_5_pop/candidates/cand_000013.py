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

    # Distance from each truck to depot
    all_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Identify index of current truck (exact match with tolerance)
    diff = truck_positions - current_position
    idx = np.where(np.all(np.isclose(diff, 0), axis=1))[0]
    if len(idx) == 0:
        # fallback: closest truck (should not happen)
        idx = [np.argmin(np.linalg.norm(diff, axis=1))]
    current_idx = idx[0]

    # Current TTT estimate (if all trucks return directly to depot now)
    current_ttt = np.max(all_depot_dists)

    best_customer = None
    best_ttt = float('inf')
    best_dist_to_current = float('inf')

    for i, cust in enumerate(available_customers):
        new_route = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        potential_dists = all_depot_dists.copy()
        potential_dists[current_idx] = new_route
        candidate_ttt = np.max(potential_dists)
        dist_to_current = np.linalg.norm(current_position - cust)

        if (candidate_ttt < best_ttt) or (candidate_ttt == best_ttt and dist_to_current < best_dist_to_current):
            best_ttt = candidate_ttt
            best_dist_to_current = dist_to_current
            best_customer = i

    # Wait if no customer improves upon current TTT
    if best_customer is None or best_ttt >= current_ttt:
        return None

    return best_customer