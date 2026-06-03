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
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # Current direct distances to depot for all trucks
    direct_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = direct_dists.max()
    active_direct = direct_dists[active_idx]

    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)

    best_index = None
    best_savings = -np.inf
    best_active_cost = np.inf
    fallback_index = None
    fallback_new_max = np.inf
    best_fallback_cost = np.inf

    for i in range(len(available_customers)):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + depot_dists[i]
        active_new = active_cost  # total distance if active serves then returns to depot

        if n_trucks == 1:
            # Single truck: just minimize total distance
            if active_cost < best_active_cost:
                best_index = i
                best_active_cost = active_cost
            continue

        # Compute costs for other trucks
        other_costs = []
        for j in range(n_trucks):
            if j == active_idx:
                continue
            other_cost = np.linalg.norm(truck_positions[j] - cust) + depot_dists[i]
            other_costs.append(other_cost)
        min_other = min(other_costs)
        other_max = np.max(np.delete(direct_dists, active_idx))
        new_max = max(active_new, other_max)

        # Check if assignment does not increase max
        if new_max <= current_max + 1e-9:
            # Eligible: does not increase max
            savings = min_other - active_cost  # positive if active is better
            if savings > best_savings or (savings == best_savings and active_cost < best_active_cost):
                best_savings = savings
                best_index = i
                best_active_cost = active_cost
        else:
            # Falls into the category that increases max; consider as fallback to minimize increase
            if new_max < fallback_new_max or (new_max == fallback_new_max and active_cost < best_fallback_cost):
                fallback_new_max = new_max
                fallback_index = i
                best_fallback_cost = active_cost

    if best_index is not None:
        return best_index
    elif fallback_index is not None:
        # Only use fallback if waiting does not yield better max
        # Waiting keeps max = current_max, so if fallback_new_max > current_max, waiting is better? But we need to serve eventually.
        # For simplicity, we always assign if there is any customer, because waiting may never lead to assignment.
        # However, to strictly minimize makespan, we could compare to waiting (return None). Here we choose to assign.
        return fallback_index
    else:
        return None