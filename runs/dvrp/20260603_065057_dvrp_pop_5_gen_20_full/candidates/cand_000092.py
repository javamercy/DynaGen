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
    # Compute other trucks' direct distance to depot (estimate of remaining tour)
    other_dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # For the current truck, we will compute tour length including candidate customer
    best_max = np.inf
    best_customer_idx = -1
    best_current_cost = np.inf
    for i in range(available_customers.shape[0]):
        cust = available_customers[i]
        current_cost = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        # Maximum tour length if this customer is assigned to current truck
        # (other trucks are unaffected at this moment)
        max_tour = max(current_cost, np.max(other_dist_to_depot))
        if max_tour < best_max or (max_tour == best_max and current_cost < best_current_cost):
            best_max = max_tour
            best_customer_idx = i
            best_current_cost = current_cost
    return best_customer_idx