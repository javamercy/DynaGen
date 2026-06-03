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
    d_curr = np.linalg.norm(available_customers - current_position, axis=1)
    n_trucks = truck_positions.shape[0]
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    if n_trucks > 1:
        all_dists = np.linalg.norm(truck_positions[:, np.newaxis, :] - available_customers, axis=2)
        all_dists[current_idx, :] = np.inf
        min_other_dist = np.min(all_dists, axis=0)
    else:
        min_other_dist = np.full(n_cust, 1e9)
    regret = d_curr - min_other_dist
    best_idx = np.argmin(regret)
    best_regret = regret[best_idx]
    upper_quartile = np.percentile(d_curr, 75)
    if best_regret > upper_quartile:
        return None
    return int(best_idx)