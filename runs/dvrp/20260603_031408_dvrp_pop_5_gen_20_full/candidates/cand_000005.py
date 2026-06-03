def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    # Convert to numpy arrays with proper shape
    cur = np.asarray(current_position).reshape(2)
    depot = np.asarray(depot_position).reshape(2)
    trucks = np.asarray(truck_positions)  # shape (n_trucks, 2)
    customers = np.asarray(available_customers)  # shape (n_available, 2)

    # Euclidean distances
    def dist(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Distance from current truck to depot
    cur_to_depot = dist(cur, depot)

    # Find index of current truck in truck_positions (by closest match)
    diffs = trucks - cur
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    cur_idx = np.argmin(dists)

    # Precompute distances from each truck to depot
    truck_to_depot = np.sqrt(np.sum((trucks - depot) ** 2, axis=1))

    # For each customer, compute marginal increases
    n_cust = len(customers)
    regret = np.empty(n_cust)
    for i in range(n_cust):
        cust = customers[i]
        # Marginal increase for current truck
        curr_inc = dist(cur, cust) + dist(cust, depot) - cur_to_depot
        # Marginal increase for best other truck
        other_inc = np.inf
        for j in range(len(trucks)):
            if j == cur_idx:
                continue
            inc = dist(trucks[j], cust) + dist(cust, depot) - truck_to_depot[j]
            if inc < other_inc:
                other_inc = inc
        regret[i] = curr_inc - other_inc

    min_regret = np.min(regret)
    if min_regret > 0:
        # Current truck is not best for any customer; wait
        return None
    else:
        # Return index of customer with minimal regret (ties break arbitrarily)
        return int(np.argmin(regret))