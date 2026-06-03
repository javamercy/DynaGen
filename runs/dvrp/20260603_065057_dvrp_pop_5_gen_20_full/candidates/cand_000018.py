import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)

    n_trucks = truck_positions.shape[0]
    n_cust = available_customers.shape[0]

    d_curr = np.linalg.norm(available_customers - current_position, axis=1)
    d_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    if n_trucks > 1:
        all_dists = np.linalg.norm(truck_positions[:, np.newaxis, :] - available_customers, axis=2)
        all_dists[current_idx, :] = np.inf
        d_other_min = np.min(all_dists, axis=0)
    else:
        d_other_min = np.full(n_cust, 1e9)

    # condition: current truck is closest (or tied) for a customer
    competitive = d_curr <= d_other_min
    if not np.any(competitive):
        return None

    epsilon = 1e-6
    # compute scores only for competitive customers; set others to inf
    scores = np.full(n_cust, np.inf)
    scores[competitive] = d_curr[competitive] * d_depot[competitive] / (d_other_min[competitive] + epsilon)
    best_idx = np.argmin(scores)
    return int(best_idx)