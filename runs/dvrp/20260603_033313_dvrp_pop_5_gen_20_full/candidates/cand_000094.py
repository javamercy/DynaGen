import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = truck_positions.shape[0]
    if n_trucks == 1:
        dists = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(dists))
    # Compute distances from each truck to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Find index of current truck (closest position)
    idx_self = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    best_idx = -1
    best_max_return = np.inf
    best_self_return = np.inf
    for i, cust in enumerate(available_customers):
        self_dist = np.linalg.norm(current_position - cust)
        cust_to_depot = np.linalg.norm(cust - depot_position)
        self_return = self_dist + cust_to_depot
        # Other trucks' return if they go directly to depot from current positions
        other_returns = truck_to_depot.copy()
        other_returns[idx_self] = -np.inf  # ignore self
        max_return = max(self_return, np.max(other_returns))
        if max_return < best_max_return or (max_return == best_max_return and self_return < best_self_return):
            best_max_return = max_return
            best_self_return = self_return
            best_idx = i
    return best_idx