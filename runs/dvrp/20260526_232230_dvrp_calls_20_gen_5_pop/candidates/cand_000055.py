import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    # distances from current truck to each customer
    d_current = np.linalg.norm(available_customers - current_position, axis=1)

    # distances from other trucks to each customer
    other_positions = truck_positions
    # shape: (n_other, n_cust)
    d_others = np.linalg.norm(available_customers[np.newaxis, :, :] - other_positions[:, np.newaxis, :], axis=2)
    min_other_dist = np.min(d_others, axis=0) if other_positions.shape[0] > 0 else np.full(d_current.shape, np.inf)

    # current truck is strictly the closest (no tie)
    is_closest = d_current < min_other_dist

    if not np.any(is_closest):
        return None

    candidate_indices = np.where(is_closest)[0]
    candidate_customers = available_customers[candidate_indices]

    # compute new return times for candidates
    d_truck_cust = d_current[candidate_indices]
    d_cust_depot = np.linalg.norm(candidate_customers - depot_position, axis=1)
    new_returns = d_truck_cust + d_cust_depot

    # max return among other trucks (current return times to depot)
    other_returns = np.linalg.norm(other_positions - depot_position, axis=1) if other_positions.shape[0] > 0 else np.array([])
    max_other_return = np.max(other_returns) if other_returns.size > 0 else 0.0

    # new max return for each candidate
    new_max = np.maximum(new_returns, max_other_return)

    # pick best: minimize new_max, tie-break min new_return
    best_local_idx = np.lexsort((new_returns, new_max))[0]
    best_global_idx = candidate_indices[best_local_idx]

    return int(best_global_idx)