import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # identify current truck index
    dists_to_cur = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dists_to_cur)
    # compute distances to other trucks
    mask = np.ones(truck_positions.shape[0], dtype=bool)
    mask[current_idx] = False
    other_trucks = truck_positions[mask]
    n_cust = available_customers.shape[0]
    scores = np.zeros(n_cust)
    d_currents = np.zeros(n_cust)
    for i in range(n_cust):
        cust = available_customers[i]
        d_cur = np.linalg.norm(cust - current_position)
        d_depot = np.linalg.norm(cust - depot_position)
        if other_trucks.shape[0] > 0:
            d_other = np.min(np.linalg.norm(other_trucks - cust, axis=1))
        else:
            d_other = 0.0
        scores[i] = d_cur + d_depot - 0.5 * d_other
        d_currents[i] = d_cur
    # find best index: min score, tie-break by min d_cur
    min_score = np.min(scores)
    candidates = np.where(scores == min_score)[0]
    if len(candidates) > 1:
        best_idx = candidates[np.argmin(d_currents[candidates])]
    else:
        best_idx = candidates[0]
    return int(best_idx)