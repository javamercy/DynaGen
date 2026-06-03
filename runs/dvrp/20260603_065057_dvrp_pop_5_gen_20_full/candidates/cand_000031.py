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
    truck_dists = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(truck_dists)
    if n_trucks > 1:
        all_dists = np.linalg.norm(truck_positions[:, np.newaxis, :] - available_customers, axis=2)
        all_dists[current_idx, :] = np.inf
        min_other_dist = np.min(all_dists, axis=0)
    else:
        min_other_dist = np.full(n_cust, 1e9)
    regret = d_curr - min_other_dist
    d_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    alpha = 0.1
    adjusted_regret = regret + alpha * d_depot
    best_idx = np.argmin(adjusted_regret)
    best_adj_regret = adjusted_regret[best_idx]
    truck_totals = np.linalg.norm(truck_positions[:, np.newaxis, :] - available_customers, axis=2).sum(axis=1)
    curr_total = truck_totals[current_idx]
    mean_total = np.mean(truck_totals)
    if mean_total > 0:
        balance_ratio = curr_total / mean_total
    else:
        balance_ratio = 1.0
    median_d_curr = np.median(d_curr)
    threshold = median_d_curr / balance_ratio if balance_ratio > 0 else median_d_curr
    if best_adj_regret > threshold:
        return None
    return int(best_idx)