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

    # distances to depot
    depot_dist = np.linalg.norm(available_customers - depot_position, axis=1)
    # active truck costs (total travel to serve customer and return)
    active_costs = np.linalg.norm(current_position - available_customers, axis=1) + depot_dist

    # compute current distances to depot for all trucks
    truck_depot_dist = np.linalg.norm(truck_positions - depot_position, axis=1)
    # max distance among other trucks
    other_max = np.max(np.delete(truck_depot_dist, active_idx))
    # if no other trucks, set a very large threshold
    if n_trucks == 1:
        other_max = np.inf

    # compute costs for other trucks (min over other trucks per customer)
    other_costs_all = []
    for j in range(n_trucks):
        if j == active_idx:
            continue
        cost = np.linalg.norm(truck_positions[j] - available_customers, axis=1) + depot_dist
        other_costs_all.append(cost)
    if other_costs_all:
        other_costs_min = np.min(other_costs_all, axis=0)
    else:
        other_costs_min = np.full(len(available_customers), np.inf)

    best_idx = None
    best_savings = -np.inf
    best_active_cost = np.inf

    for i in range(len(available_customers)):
        if active_costs[i] <= other_max:  # qualifies: does not exceed current max other distance
            savings = other_costs_min[i] - active_costs[i] if other_costs_min[i] < np.inf else 0
            if savings > best_savings or (savings == best_savings and active_costs[i] < best_active_cost):
                best_savings = savings
                best_active_cost = active_costs[i]
                best_idx = i

    return best_idx