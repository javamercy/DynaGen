import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_cust = available_customers.shape[0]
    n_trucks = truck_positions.shape[0]
    d_curr = np.linalg.norm(available_customers - current_position, axis=1)
    dists_to_curr = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dists_to_curr)
    if n_trucks > 1:
        all_dists = np.linalg.norm(truck_positions[:, np.newaxis, :] - available_customers, axis=2)
        all_dists[current_idx, :] = np.inf
        d_other = np.min(all_dists, axis=0)
    else:
        d_other = np.full(n_cust, 1e9)
    regret = d_curr - d_other
    d_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    truck_depot_dist = np.linalg.norm(current_position - depot_position)
    alpha = 0.2
    beta = 0.1
    score = regret + alpha * d_depot + beta * truck_depot_dist
    best_idx = np.argmin(score)
    best_score = score[best_idx]
    threshold = 0.2 * np.median(d_curr) + 0.1 * truck_depot_dist
    if best_score > threshold:
        return None
    return int(best_idx)