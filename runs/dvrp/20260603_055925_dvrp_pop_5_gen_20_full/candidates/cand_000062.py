import numpy as np

def choose_next_customer(current_position: np.ndarray, depot_position: np.ndarray, truck_positions: np.ndarray, available_customers: np.ndarray) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    n_trucks = truck_positions.shape[0]
    # identify current truck index
    current_mask = np.all(np.isclose(truck_positions, current_position), axis=1)
    other_trucks = truck_positions[~current_mask]

    # distances from current truck to each customer
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # new return time for current truck if it goes to customer
    new_return_current = curr_to_cust + cust_to_depot

    if other_trucks.shape[0] == 0:
        # single truck: just minimize own return time
        best_idx = np.argmin(new_return_current)
        return int(best_idx)

    # heuristic: other trucks' direct return times (if they go straight to depot now)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    max_other = np.max(other_depot_dists)

    # new overall max if current truck serves customer i
    new_max = np.maximum(new_return_current, max_other)
    best_idx = np.argmin(new_max)
    best_new_max = new_max[best_idx]

    # wait if best customer causes more than 20% increase over current max other return
    if best_new_max > 1.2 * max_other:
        return None
    else:
        return int(best_idx)