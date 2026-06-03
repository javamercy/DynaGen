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

    # Precompute depot distances for all customers
    delta = available_customers - depot_position
    depot_dist = np.linalg.norm(delta, axis=1)

    best = None
    best_rel_savings = -np.inf
    best_active_cost = np.inf

    for i in range(len(available_customers)):
        active_cost = np.linalg.norm(current_position - available_customers[i]) + depot_dist[i]

        # Compute costs for other trucks
        other_costs = []
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - available_customers[i]) + depot_dist[i]
            other_costs.append(cost)

        if n_trucks == 1:
            if active_cost < best_active_cost:
                best = i
                best_active_cost = active_cost
        else:
            min_other = min(other_costs)
            if active_cost <= min_other:
                rel_savings = (min_other - active_cost) / (active_cost + 1e-12)  # avoid division by zero
                if rel_savings > best_rel_savings or (rel_savings == best_rel_savings and active_cost < best_active_cost):
                    best_rel_savings = rel_savings
                    best = i
                    best_active_cost = active_cost

    if best is None:
        return None
    return best