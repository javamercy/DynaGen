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

    # current maximum return distance among all trucks
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_return = np.max(truck_depot_dists)

    beta = 0.5  # penalty weight

    best_regret = -np.inf
    best_index = None

    for i in range(len(available_customers)):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)

        # single truck case
        if n_trucks == 1:
            if active_cost < best_regret:  # reuse best_regret as min cost
                best_regret = active_cost
                best_index = i
            continue

        # compute min cost among other trucks
        min_other = np.inf
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            if cost < min_other:
                min_other = cost

        # penalty: increase in max return if this customer is served
        new_max = max(current_max_return, active_cost)
        penalty = max(0, new_max - current_max_return)

        regret = (min_other - active_cost) - beta * penalty

        if regret > best_regret:
            best_regret = regret
            best_index = i

    if best_index is not None and best_regret > 0:
        return best_index
    else:
        return None