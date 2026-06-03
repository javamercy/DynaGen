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
    # distances to depot for all trucks
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # find active truck index
    active_idx = int(np.flatnonzero(np.all(truck_positions == current_position, axis=1))[0])
    cur_dist = truck_to_depot[active_idx]

    # compute max of other trucks' depot distances
    if n_trucks == 1:
        # single truck: always serve with smallest new return time
        dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
        cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
        new_return = dist_to_cust + cust_to_depot
        best_idx = int(np.argmin(new_return))
        return best_idx

    others_mask = np.ones(n_trucks, dtype=bool)
    others_mask[active_idx] = False
    max_other = float(np.max(truck_to_depot[others_mask]))

    # for each customer
    dist_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    new_return = dist_to_cust + cust_to_depot

    if cur_dist <= max_other + 1e-12:  # active is not sole max (within tolerance)
        # find customers that do not increase makespan: new_return <= max_other
        ok = new_return <= max_other + 1e-12
        if np.any(ok):
            # among those, minimize new_return
            best_candidates = np.where(ok)[0]
            best_idx = int(best_candidates[np.argmin(new_return[ok])])
            return best_idx
        else:
            # all customers would increase makespan; wait
            return None
    else:  # active is the maximum
        # serve the customer with smallest new return
        best_idx = int(np.argmin(new_return))
        return best_idx