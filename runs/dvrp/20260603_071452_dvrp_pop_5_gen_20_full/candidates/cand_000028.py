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

    delta = available_customers - depot_position
    depot_dist = np.linalg.norm(delta, axis=1)

    active_to_cust = np.linalg.norm(current_position - available_customers, axis=1)
    active_cost = active_to_cust + depot_dist

    if n_trucks == 1:
        best_idx = np.argmin(active_cost)
        return int(best_idx)

    best_idx = None
    best_advantage = -np.inf
    best_active_cost = np.inf

    for i in range(len(available_customers)):
        # compute min other cost
        other_costs = []
        for j in range(n_trucks):
            if j == active_idx:
                continue
            cost = np.linalg.norm(truck_positions[j] - available_customers[i]) + depot_dist[i]
            other_costs.append(cost)
        min_other = min(other_costs)

        if active_cost[i] <= min_other:
            advantage = min_other - active_cost[i]
            if advantage > best_advantage or (advantage == best_advantage and active_cost[i] < best_active_cost):
                best_advantage = advantage
                best_active_cost = active_cost[i]
                best_idx = i

    return best_idx