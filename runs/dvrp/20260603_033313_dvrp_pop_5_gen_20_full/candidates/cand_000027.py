import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    n_cust = available_customers.shape[0]
    if n_cust == 0:
        return None
    # Identify active truck index
    dists_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(dists_to_trucks)
    # Precompute distances from each truck to depot
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    best_idx = None
    best_makespan = float('inf')
    for i, cust in enumerate(available_customers):
        # Distances from all trucks to this customer
        dists = np.linalg.norm(truck_positions - cust, axis=1)
        min_dist = np.min(dists)
        # Check if active truck is a nearest truck (unique by tie-breaking with smallest index)
        if np.isclose(dists[active_idx], min_dist) and active_idx == np.argmin(dists):
            # Estimated return time if active truck serves this customer
            return_active = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
            # Maximum return time among other trucks if they go directly to depot
            max_other_return = np.max(np.delete(depot_dists, active_idx))
            estimated_makespan = max(return_active, max_other_return)
            if estimated_makespan < best_makespan:
                best_makespan = estimated_makespan
                best_idx = i
    return best_idx