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

    depot_dist = np.linalg.norm(available_customers - depot_position, axis=1)
    active_costs = np.linalg.norm(current_position - available_customers, axis=1) + depot_dist

    other_costs_list = []
    for j in range(n_trucks):
        if j == active_idx:
            continue
        cost = np.linalg.norm(truck_positions[j] - available_customers, axis=1) + depot_dist
        other_costs_list.append(cost)
    if not other_costs_list:
        best_idx = int(np.argmin(active_costs))
        return best_idx

    other_costs_min = np.min(other_costs_list, axis=0)
    other_max_depot = max(
        np.linalg.norm(truck_positions[j] - depot_position)
        for j in range(n_trucks) if j != active_idx
    )
    penalty_factor = 0.5
    effective_active = active_costs + penalty_factor * np.maximum(0, active_costs - other_max_depot)

    best_candidate = None
    best_savings = -np.inf
    best_effective = np.inf
    for i, eff in enumerate(effective_active):
        if eff <= other_costs_min[i]:
            savings = other_costs_min[i] - eff
            if savings > best_savings or (savings == best_savings and eff < best_effective):
                best_savings = savings
                best_effective = eff
                best_candidate = i

    if best_candidate is not None:
        return best_candidate

    threshold = 1.1
    fallback_candidate = None
    fallback_eff = np.inf
    for i, eff in enumerate(effective_active):
        if eff <= other_costs_min[i] * threshold:
            if eff < fallback_eff:
                fallback_eff = eff
                fallback_candidate = i

    return fallback_candidate if fallback_candidate is not None else None