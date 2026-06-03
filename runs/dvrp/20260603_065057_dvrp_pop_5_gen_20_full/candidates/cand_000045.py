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
    if n_trucks == 1:
        min_other_dist = np.full(n_cust, 1e9)
    else:
        # distances from all trucks to all customers
        all_dists = np.linalg.norm(truck_positions[:, np.newaxis, :] - available_customers, axis=2)
        # find index of current truck by closest position
        diff = np.linalg.norm(truck_positions - current_position, axis=1)
        current_idx = np.argmin(diff)
        all_dists[current_idx, :] = np.inf
        min_other_dist = np.min(all_dists, axis=0)
    regret = d_curr - min_other_dist
    best_idx = np.argmin(regret)
    best_regret = regret[best_idx]
    median_d_curr = np.median(d_curr)
    if best_regret > 1.5 * median_d_curr:
        return None
    return int(best_idx)