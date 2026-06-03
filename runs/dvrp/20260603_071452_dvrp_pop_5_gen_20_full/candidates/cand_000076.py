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
    # Identify active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)

    # Active truck's cost to each customer
    active_travel = np.linalg.norm(current_position - available_customers, axis=1)
    active_cost = active_travel + depot_dists

    # Single truck case
    if n_trucks == 1:
        best_idx = np.argmin(active_cost)
        return int(best_idx)

    # Compute other trucks' costs
    other_positions = np.delete(truck_positions, active_idx, axis=0)
    other_dists = np.linalg.norm(
        available_customers[:, np.newaxis, :] - other_positions[np.newaxis, :, :], axis=2
    )
    other_costs = other_dists + depot_dists[:, np.newaxis]
    min_other = np.min(other_costs, axis=1)
    savings = min_other - active_cost

    # Fixed slack = 10%
    slack = 0.1
    eligible = active_cost <= (1 + slack) * min_other

    if not np.any(eligible):
        return None

    eligible_indices = np.where(eligible)[0]
    eligible_savings = savings[eligible]
    eligible_active = active_cost[eligible]
    # lexsort: (active asc, savings desc) -> need negative savings
    order = np.lexsort((eligible_active, -eligible_savings))
    best_eligible_idx = eligible_indices[order[0]]
    return int(best_eligible_idx)