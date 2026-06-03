import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    n_trucks = len(truck_positions)
    # find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # current return times if trucks go directly to depot
    direct_returns = np.linalg.norm(truck_positions - depot_position, axis=1)
    active_direct = direct_returns[active_idx]
    # max return among other trucks
    other_returns = np.delete(direct_returns, active_idx)
    max_other_return = np.max(other_returns) if len(other_returns) > 0 else -np.inf

    # distances to customers
    dist_to_cust = np.linalg.norm(current_position - available_customers, axis=1)
    depot_dist = np.linalg.norm(available_customers - depot_position, axis=1)
    active_new_returns = dist_to_cust + depot_dist

    # new max if active truck serves each customer
    new_maxes = np.maximum(max_other_return, active_new_returns)

    # best other truck's new return for each customer
    best_other_new = np.full(len(available_customers), np.inf)
    for j in range(n_trucks):
        if j == active_idx:
            continue
        other_new = np.linalg.norm(truck_positions[j] - available_customers, axis=1) + depot_dist
        best_other_new = np.minimum(best_other_new, other_new)

    # candidate selection: minimize new_max, tie-break by active_new_return
    min_new_max = np.min(new_maxes)
    candidates = np.where(new_maxes == min_new_max)[0]
    if len(candidates) == 0:
        return None
    # among candidates, choose the one with smallest active_new_return
    best_idx = candidates[np.argmin(active_new_returns[candidates])]

    # threshold condition: active cost > 1.1 * best other cost => wait
    threshold = 1.1
    if len(other_returns) > 0 and active_new_returns[best_idx] > threshold * best_other_new[best_idx]:
        return None

    return int(best_idx)