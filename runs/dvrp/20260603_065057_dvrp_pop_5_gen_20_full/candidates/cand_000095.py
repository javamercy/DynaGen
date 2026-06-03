import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    # identify current truck index
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # compute truck distances to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    mean_truck_to_depot = np.mean(truck_to_depot)
    current_remoteness = truck_to_depot[current_idx] / (mean_truck_to_depot + 1e-6)
    # fixed alpha
    alpha = 0.1
    # beta for depot pressure
    beta = 0.05
    best_score = -np.inf
    best_idx = -1
    for i, cust in enumerate(available_customers):
        this_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        other_costs = [
            np.linalg.norm(truck_positions[j] - cust) + np.linalg.norm(cust - depot_position)
            for j in range(n_trucks) if j != current_idx
        ]
        if other_costs:
            other_min = min(other_costs)
            regret = max(0, other_min - this_cost)
        else:
            regret = 0
        # depot pressure: penalize customers far from depot more when truck is remote
        depot_pressure = np.linalg.norm(cust - depot_position) * current_remoteness
        score = regret - alpha * this_cost - beta * depot_pressure
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx if best_idx != -1 else None