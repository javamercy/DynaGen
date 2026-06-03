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
    avail = available_customers
    n_avail = len(avail)
    n_trucks = truck_positions.shape[0]
    best_idx = None
    best_score = -np.inf
    for i in range(n_avail):
        d_cur = np.linalg.norm(cur - avail[i])
        d_depot = np.linalg.norm(depot_position - avail[i])
        # compute distance to nearest other truck
        dists_to_trucks = np.linalg.norm(truck_positions - avail[i], axis=1)
        sorted_dists = np.sort(dists_to_trucks)
        # check if the closest truck is this current truck
        if n_trucks > 1 and np.abs(sorted_dists[0] - d_cur) < 1e-6:
            min_other = sorted_dists[1]
        else:
            min_other = sorted_dists[0] if n_trucks > 1 else 0.0
        score = -d_depot - d_cur + 0.5 * min_other
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx