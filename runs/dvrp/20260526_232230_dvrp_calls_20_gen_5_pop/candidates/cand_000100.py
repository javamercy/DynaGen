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

    depot = depot_position
    # identify active truck index
    diffs = truck_positions - current_position
    active_idx = np.argmin(np.einsum('ij,ij->i', diffs, diffs))
    # current return times for all trucks (if they returned now)
    all_returns = dist(truck_positions, depot)
    current_max_return = np.max(all_returns)
    # other trucks' returns (excluding active)
    other_returns = np.delete(all_returns, active_idx)
    other_max_return = np.max(other_returns) if other_returns.size > 0 else 0.0

    # customer distances
    d_current_cust = dist(current_position, available_customers)
    d_cust_depot = dist(available_customers, depot)
    new_returns = d_current_cust + d_cust_depot

    # compute new max and gap penalty for each customer
    new_max = np.maximum(new_returns, other_max_return)
    gap = np.maximum(new_returns - other_max_return, 0.0)
    penalty_weight = 1.0
    scores = new_max + penalty_weight * gap

    # waiting decision: if the smallest new max exceeds current_max by >10%, wait
    best_new_max = np.min(new_max)
    if best_new_max > 1.1 * current_max_return:
        return None

    # choose best customer based on scores, tie-break by new_return
    min_score = np.min(scores)
    candidates = np.where(scores == min_score)[0]
    if len(candidates) == 1:
        best_idx = candidates[0]
    else:
        best_idx = candidates[np.argmin(new_returns[candidates])]
    return int(best_idx)