import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    cur = current_position
    depot = depot_position
    avail = available_customers
    trucks = truck_positions
    best_score = -np.inf
    best_idx = None
    n = len(avail)
    for i in range(n):
        cust = avail[i]
        d_cur = np.linalg.norm(cur - cust)
        d_depot = np.linalg.norm(cust - depot)
        # distance to nearest other truck (not the current one)
        dists = np.linalg.norm(trucks - cust, axis=1)
        sorted_dists = np.sort(dists)
        if len(sorted_dists) >= 2:
            nearest_other = sorted_dists[1]  # second smallest, smallest is 0 from current
        else:
            nearest_other = 0.0  # only one truck
        score = nearest_other - (d_cur + d_depot)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx