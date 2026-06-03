def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))
    # Find index of current truck
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    # Distances from current truck to customers
    d_cur = np.linalg.norm(available_customers - current_position, axis=1)
    # Distances from customers to depot
    d_cust_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # Current truck's total cost to serve and return
    current_costs = d_cur + d_cust_depot
    # Best other truck's cost
    best_other_costs = np.full(len(available_customers), np.inf)
    for i, cust in enumerate(available_customers):
        dists_to_trucks = np.linalg.norm(truck_positions - cust, axis=1)
        other_dists = np.delete(dists_to_trucks, current_truck_idx)
        min_other = np.min(other_dists) if len(other_dists) > 0 else float('inf')
        other_cost = min_other + d_cust_depot[i]
        best_other_costs[i] = other_cost
    net_gains = best_other_costs - current_costs
    max_net_gain = np.max(net_gains)
    if max_net_gain > 0:
        return int(np.argmax(net_gains))
    else:
        return None