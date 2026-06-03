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
    # active truck cost = distance to customer + distance from customer to depot
    active_costs = np.linalg.norm(current_position - available_customers, axis=1) + depot_dist

    # special case: only one truck
    if n_trucks == 1:
        best_idx = int(np.argmin(active_costs))
        return best_idx

    # compute costs for other trucks (min over other trucks per customer)
    other_costs_all = []
    for j in range(n_trucks):
        if j == active_idx:
            continue
        cost = np.linalg.norm(truck_positions[j] - available_customers, axis=1) + depot_dist
        other_costs_all.append(cost)
    other_costs_min = np.min(other_costs_all, axis=0)

    # current max return time if all trucks return directly
    truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_rt = np.max(truck_depot_dists)

    # first pass: customers where active is strictly best (active_cost <= other_min)
    best_candidate = None
    best_savings = -np.inf
    best_active_cost = np.inf
    for i, active_cost in enumerate(active_costs):
        if active_cost <= other_costs_min[i]:
            savings = other_costs_min[i] - active_cost
            if savings > best_savings or (savings == best_savings and active_cost < best_active_cost):
                best_savings = savings
                best_active_cost = active_cost
                best_candidate = i

    if best_candidate is not None:
        return best_candidate

    # second pass: no customer where active is best; allow within 10% of other best and within 10% of current max
    fallback_candidate = None
    fallback_cost = np.inf
    threshold = 1.1
    for i, active_cost in enumerate(active_costs):
        if active_cost <= other_costs_min[i] * threshold and active_cost <= current_max_rt * threshold:
            if active_cost < fallback_cost:
                fallback_cost = active_cost
                fallback_candidate = i

    return fallback_candidate if fallback_candidate is not None else None