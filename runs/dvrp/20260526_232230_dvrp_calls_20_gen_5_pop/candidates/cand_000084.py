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

    # distances helper
    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    depot = depot_position
    # find index of active truck (the one at current_position)
    # truck_positions shape (n_trucks, 2)
    diffs = truck_positions - current_position
    active_idx = np.argmin(np.einsum('ij,ij->i', diffs, diffs))  # squared norm minimization
    # all truck returns to depot
    all_returns = dist(truck_positions, depot)
    # other trucks' returns
    other_returns = np.delete(all_returns, active_idx)
    other_max_return = np.max(other_returns) if other_returns.size > 0 else 0.0

    # customer distances
    d_current_cust = dist(current_position, available_customers)  # (n_available,)
    d_cust_depot = dist(available_customers, depot)  # (n_available,)
    new_returns = d_current_cust + d_cust_depot

    # compute new_max and penalty
    new_max = np.maximum(new_returns, other_max_return)
    gap = np.maximum(new_returns - other_max_return, 0.0)
    penalty_weight = 0.5
    scores = new_max + penalty_weight * gap

    # tie-break by new_return
    # find minimal score; if tie, smallest new_return
    min_score = np.min(scores)
    candidates = np.where(scores == min_score)[0]
    if len(candidates) == 1:
        best_idx = candidates[0]
    else:
        # among ties, pick smallest new_return
        best_candidate = candidates[np.argmin(new_returns[candidates])]
        best_idx = best_candidate

    return int(best_idx)