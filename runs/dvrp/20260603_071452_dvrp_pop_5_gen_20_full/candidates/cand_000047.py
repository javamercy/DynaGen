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

    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    best_index = None
    best_savings = -np.inf
    best_active_cost = np.inf
    fallback_index = None
    fallback_active_cost = np.inf

    # Depot distance of active truck for dynamic threshold
    active_depot_dist = np.linalg.norm(current_position - depot_position)
    max_depot_dist = np.max(np.linalg.norm(truck_positions - depot_position, axis=1))
    if max_depot_dist == 0:
        frac = 0.0
    else:
        frac = active_depot_dist / max_depot_dist
    fallback_factor = 1.0 - 0.15 * frac  # tighter for distant trucks

    for i in range(len(available_customers)):
        cust = available_customers[i]
        active_cost = np.linalg.norm(current_position - cust) + depot_dists[i]

        if n_trucks == 1:
            if active_cost < best_active_cost:
                best_index = i
                best_active_cost = active_cost
            continue

        other_costs = []
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - cust) + depot_dists[i]
            other_costs.append(cost)
        min_other = min(other_costs)

        # Primary rule: active achieves savings over best other
        if active_cost <= min_other + 1e-9:
            savings = min_other - active_cost
            if savings > best_savings or (savings == best_savings and active_cost < best_active_cost):
                best_savings = savings
                best_index = i
                best_active_cost = active_cost
        else:
            # Fallback: active cost within dynamic threshold
            if active_cost <= fallback_factor * min_other:
                if active_cost < fallback_active_cost:
                    fallback_index = i
                    fallback_active_cost = active_cost

    if best_index is not None:
        return best_index
    elif fallback_index is not None:
        return fallback_index
    else:
        return None