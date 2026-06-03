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
    # Find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    active_costs = np.linalg.norm(current_position - available_customers, axis=1) + depot_dists

    if n_trucks == 1:
        return int(np.argmin(active_costs))

    # Compute min cost among other trucks for each customer
    other_costs = np.full(len(available_customers), np.inf)
    for j in range(n_trucks):
        if j == active_idx:
            continue
        truck_cost = np.linalg.norm(truck_positions[j] - available_customers, axis=1) + depot_dists
        other_costs = np.minimum(other_costs, truck_cost)

    savings = other_costs - active_costs
    positive_mask = savings > 0

    if np.any(positive_mask):
        # Among positive savings, select max savings, tie-break by min active cost
        best_idx = None
        best_savings = -np.inf
        best_active = np.inf
        for i in range(len(available_customers)):
            if positive_mask[i]:
                if savings[i] > best_savings or (savings[i] == best_savings and active_costs[i] < best_active):
                    best_savings = savings[i]
                    best_active = active_costs[i]
                    best_idx = i
        return best_idx
    else:
        # No positive savings: decide whether to wait based on proximity to depot
        active_dist = np.linalg.norm(current_position - depot_position)
        other_dists = np.array([np.linalg.norm(truck_positions[j] - depot_position) for j in range(n_trucks) if j != active_idx])
        if len(other_dists) > 0 and active_dist > np.min(other_dists):
            # Active truck is not the closest to depot: serve the cheapest customer to avoid idling
            return int(np.argmin(active_costs))
        else:
            return None