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
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    current_idx = np.argmin(diff)
    # Precompute other trucks' distances to depot
    other_truck_to_depot = []
    for j in range(n_trucks):
        if j != current_idx:
            other_truck_to_depot.append(np.linalg.norm(truck_positions[j] - depot_position))
    min_other_to_depot = min(other_truck_to_depot) if other_truck_to_depot else 0.0
    best_makespan = np.inf
    best_customer_idx = -1
    best_current_cost = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        current_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        makespan = max(current_cost, min_other_to_depot)
        if makespan < best_makespan or (makespan == best_makespan and current_cost < best_current_cost):
            best_makespan = makespan
            best_customer_idx = i
            best_current_cost = current_cost
    return best_customer_idx