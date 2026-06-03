def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    import numpy as np
    if available_customers.shape[0] == 0:
        return None
    # identify current truck index in truck_positions
    dists = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(dists)
    # other trucks mask
    mask = np.ones(truck_positions.shape[0], dtype=bool)
    mask[current_idx] = False
    other_trucks = truck_positions[mask]
    # compute per customer
    n_cust = available_customers.shape[0]
    costs = np.zeros(n_cust)
    for i in range(n_cust):
        cust = available_customers[i]
        d_current = np.linalg.norm(cust - current_position)
        d_depot = np.linalg.norm(cust - depot_position)
        if other_trucks.shape[0] > 0:
            d_other = np.min(np.linalg.norm(other_trucks - cust, axis=1))
        else:
            d_other = 0.0  # no other trucks, ignore
        costs[i] = d_current + 0.8 * d_depot - 0.3 * d_other
    best_idx = int(np.argmin(costs))
    return best_idx