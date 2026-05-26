import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None

    n_avail = len(available_customers)
    n_trucks = len(truck_positions)

    # distances from active truck to customers and from customers to depot
    d_active = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    active_return = d_active + cust_to_depot

    # ownership: is active truck the closest to each customer?
    # compute min distance from each customer to any other truck
    # active index
    active_idx = np.where((truck_positions == current_position).all(axis=1))[0][0]
    if n_trucks == 1:
        owned_mask = np.ones(n_avail, dtype=bool)
    else:
        other_positions = np.delete(truck_positions, active_idx, axis=0)
        # distances to other trucks
        diff = available_customers[:, np.newaxis, :] - other_positions[np.newaxis, :, :]
        dist_other = np.linalg.norm(diff, axis=2)  # (n_avail, n_other)
        min_other = np.min(dist_other, axis=1)
        owned_mask = d_active <= min_other + 1e-9

    # regret: increase in TTT if active serves vs best other truck
    # current max return time of all trucks (as if they already returned)
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_return = np.max(truck_to_depot)

    increase_active = np.maximum(0, active_return - current_max_return)

    # best other return for each customer
    if n_trucks == 1:
        best_other_return = np.full(n_avail, np.inf)
    else:
        # distances from each other truck to each customer
        # we already have dist_other; compute return for each other truck: dist_other + cust_to_depot[:, None]
        other_returns = dist_other + cust_to_depot[:, np.newaxis]  # (n_avail, n_other)
        # best among other trucks (min return)
        best_other_return = np.min(other_returns, axis=1)

    increase_other = np.maximum(0, best_other_return - current_max_return)
    regret = increase_active - increase_other
    regret_valid = regret <= 0

    # combine: customer must be owned or have non-positive regret (or both)
    valid = owned_mask | regret_valid
    if not np.any(valid):
        return None

    candidates = np.where(valid)[0]
    # among candidates, pick the one minimizing active's immediate return time
    best_idx = candidates[np.argmin(active_return[candidates])]
    return int(best_idx)