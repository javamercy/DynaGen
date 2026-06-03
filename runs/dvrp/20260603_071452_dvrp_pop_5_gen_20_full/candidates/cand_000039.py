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

    # current distances to depot for all trucks
    current_return_times = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(current_return_times)

    # depot distances for customers
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)

    # active costs
    active_costs = np.linalg.norm(current_position - available_customers, axis=1) + depot_dists

    # other min costs
    other_costs = np.full(len(available_customers), np.inf)
    for j in range(n_trucks):
        if j == active_idx:
            continue
        truck_cost = np.linalg.norm(truck_positions[j] - available_customers, axis=1) + depot_dists
        other_costs = np.minimum(other_costs, truck_cost)

    # candidate mask: active cost <= other cost
    mask = active_costs <= other_costs
    if not np.any(mask):
        return None

    # among candidates, select one that minimizes new_max = max(active_cost, current_max)
    best_idx = None
    best_new_max = np.inf
    best_regret = -np.inf
    best_active = np.inf
    for i in range(len(available_customers)):
        if not mask[i]:
            continue
        new_max = max(active_costs[i], current_max)
        regret = other_costs[i] - active_costs[i]
        # prefer smaller new_max, then larger regret, then smaller active cost
        if new_max < best_new_max or (new_max == best_new_max and regret > best_regret) or (new_max == best_new_max and regret == best_regret and active_costs[i] < best_active):
            best_new_max = new_max
            best_regret = regret
            best_active = active_costs[i]
            best_idx = i
    return best_idx