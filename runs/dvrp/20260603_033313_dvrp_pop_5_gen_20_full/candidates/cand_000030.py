import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    
    # Distances from active truck to each customer
    active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # Distances from each customer to depot
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # Active truck's total route if it serves customer then returns to depot
    active_total = active_to_cust + cust_to_depot
    
    # Distances from all trucks to depot
    truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Identify active truck index
    is_active = np.all(np.isclose(truck_positions, current_position), axis=1)
    active_idx = np.where(is_active)[0][0]
    # Maximum distance from depot among other trucks
    other_max_depot = np.max(np.delete(truck_to_depot, active_idx))
    
    # For each customer, compute the max of active_total and other_max_depot
    candidate_max = np.maximum(active_total, other_max_depot)
    # Choose customer that minimizes this max, break ties by smallest active_total
    best_idx = np.argmin(candidate_max)
    # If tie, already broken by argmin (first occurrence)
    
    return int(best_idx)