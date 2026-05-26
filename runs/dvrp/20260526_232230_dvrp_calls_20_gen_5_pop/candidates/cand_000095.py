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
    # find active truck index
    diffs = truck_positions - current_position
    active_idx = np.argmin(np.einsum('ij,ij->i', diffs, diffs))

    # all truck return distances to depot
    all_returns = dist(truck_positions, depot)
    current_return = all_returns[active_idx]
    other_returns = np.delete(all_returns, active_idx)
    other_max = np.max(other_returns) if other_returns.size > 0 else 0.0
    current_max = np.max(all_returns)

    # customer distances
    d_current_cust = dist(current_position, available_customers)
    d_cust_depot = dist(available_customers, depot)
    new_returns = d_current_cust + d_cust_depot

    # compute new max and gap penalty
    new_max = np.maximum(new_returns, other_max)
    gap = np.maximum(new_returns - other_max, 0.0)
    penalty_weight = 1.0
    scores = new_max + penalty_weight * gap

    # pick best by min score, tie-break by new return
    min_score = np.min(scores)
    candidates = np.where(scores == min_score)[0]
    if len(candidates) == 1:
        best_idx = candidates[0]
    else:
        best_idx = candidates[np.argmin(new_returns[candidates])]

    # waiting logic: if serving best customer increases max return by more than 10%, wait
    threshold = 0.1
    best_new_max = new_max[best_idx]
    if best_new_max > current_max * (1 + threshold):
        return None
    else:
        return int(best_idx)