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

    # distances from customers to depot
    depot_dist = np.linalg.norm(available_customers - depot_position, axis=1)
    # active truck costs
    active_costs = np.linalg.norm(current_position - available_customers, axis=1) + depot_dist

    # compute costs for other trucks (min over other trucks per customer)
    other_costs_all = []
    for j in range(n_trucks):
        if j == active_idx:
            continue
        cost = np.linalg.norm(truck_positions[j] - available_customers, axis=1) + depot_dist
        other_costs_all.append(cost)
    if not other_costs_all:   # only one truck
        best_idx = int(np.argmin(active_costs))
        return best_idx

    other_costs_min = np.min(other_costs_all, axis=0)  # min cost among other trucks per customer

    # First, consider customers where active is at least as good as the best other
    best_candidate = None
    best_savings = -np.inf
    best_active_cost = np.inf
    for i, active_cost in enumerate(active_costs):
        other_min = other_costs_min[i]
        if active_cost <= other_min:
            savings = other_min - active_cost
            if savings > best_savings or (savings == best_savings and active_cost < best_active_cost):
                best_savings = savings
                best_active_cost = active_cost
                best_candidate = i

    if best_candidate is not None:
        return best_candidate

    # Fallback: choose customer with smallest regret (active - other_min)
    best_regret = np.inf
    fallback_candidate = None
    for i, active_cost in enumerate(active_costs):
        other_min = other_costs_min[i]
        regret = active_cost - other_min
        if regret < best_regret or (regret == best_regret and active_cost < best_active_cost):
            best_regret = regret
            best_active_cost = active_cost
            fallback_candidate = i

    return fallback_candidate if fallback_candidate is not None else None