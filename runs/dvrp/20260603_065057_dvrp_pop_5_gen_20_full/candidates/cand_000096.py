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
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # compute distances from each other truck to depot
    other_depot_dists = [np.linalg.norm(truck_positions[j] - depot_position) for j in range(n_trucks) if j != current_idx]
    max_other = max(other_depot_dists) if other_depot_dists else 0.0
    best_ttt = np.inf
    best_idx = -1
    best_this = None
    for i, cust in enumerate(available_customers):
        d1 = np.linalg.norm(current_position - cust)
        d2 = np.linalg.norm(cust - depot_position)
        this_cost = d1 + d2
        ttt_est = max(this_cost, max_other)
        if ttt_est < best_ttt or (ttt_est == best_ttt and (best_this is None or this_cost < best_this)):
            best_ttt = ttt_est
            best_idx = i
            best_this = this_cost
    return best_idx