import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    # identify current truck index
    dists_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dists_to_current)
    # direct return distance for current truck if it does nothing
    direct_return = np.linalg.norm(current_position - depot_position)
    # compute other trucks' max direct return
    if n_trucks == 1:
        # single truck: always pick the customer with minimal estimated route
        costs = np.linalg.norm(current_position - available_customers, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1)
        return int(np.argmin(costs))
    else:
        other_mask = np.ones(n_trucks, dtype=bool)
        other_mask[current_idx] = False
        other_positions = truck_positions[other_mask]
        other_max = np.max(np.linalg.norm(other_positions - depot_position, axis=1))
        # compute costs for each customer if served by current truck
        costs = np.linalg.norm(current_position - available_customers, axis=1) + np.linalg.norm(available_customers - depot_position, axis=1)
        # find customers that do not increase the current max
        valid = np.where(costs <= other_max)[0]
        if len(valid) > 0:
            best_local_idx = np.argmin(costs[valid])
            return int(valid[best_local_idx])
        else:
            return None