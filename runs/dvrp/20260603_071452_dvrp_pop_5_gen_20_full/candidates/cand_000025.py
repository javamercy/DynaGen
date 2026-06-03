import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # distances from each customer to depot
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    # distances from each customer to each truck: shape (n_customers, n_trucks)
    diffs = available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :]
    truck_dists = np.linalg.norm(diffs, axis=2)
    costs = truck_dists + depot_dists[:, np.newaxis]
    # find active truck index
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0]
    if len(active_idx) == 0:
        raise ValueError("current_position not in truck_positions")
    active_idx = active_idx[0]
    active_costs = costs[:, active_idx]
    if len(truck_positions) == 1:
        return int(np.argmin(active_costs))
    # min cost among other trucks
    other_costs = np.delete(costs, active_idx, axis=1)
    min_other = np.min(other_costs, axis=1)
    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(min_other == 0, np.where(active_costs == 0, 1.0, np.inf), active_costs / min_other)
    eligible = ratio <= 1.2
    if not np.any(eligible):
        return None
    eligible_indices = np.where(eligible)[0]
    # sort by ratio, then active_cost
    sorted_order = np.lexsort((active_costs[eligible_indices], ratio[eligible_indices]))
    best_idx = int(eligible_indices[sorted_order[0]])
    return best_idx