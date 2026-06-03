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
    # Find index of active truck
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # Precompute depot distances for all customers
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)

    # Compute active costs
    active_costs = np.linalg.norm(current_position - available_customers, axis=1) + depot_dists

    # Compute minimum cost among other trucks for each customer
    other_costs = np.full(len(available_customers), np.inf)
    for j in range(n_trucks):
        if j == active_idx:
            continue
        truck_cost = np.linalg.norm(truck_positions[j] - available_customers, axis=1) + depot_dists
        other_costs = np.minimum(other_costs, truck_cost)

    # Identify candidates where active cost <= min other cost
    mask = active_costs <= other_costs
    if not np.any(mask):
        return None

    savings = other_costs - active_costs
    # Select candidate with max savings, tie-break by min active cost
    best_idx = None
    best_savings = -np.inf
    best_active = np.inf
    for i in range(len(available_customers)):
        if mask[i]:
            if savings[i] > best_savings or (savings[i] == best_savings and active_costs[i] < best_active):
                best_savings = savings[i]
                best_active = active_costs[i]
                best_idx = i
    return best_idx