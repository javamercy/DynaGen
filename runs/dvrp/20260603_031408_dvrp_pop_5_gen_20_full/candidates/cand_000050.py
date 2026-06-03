import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # distances from all trucks to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # identify active truck index by closest to current_position
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # other trucks' distances to depot
    others_dist = np.delete(dist_to_depot, active_idx)
    others_max = np.max(others_dist) if others_dist.size > 0 else 0.0
    best_idx = None
    best_max = float('inf')
    best_travel = float('inf')
    best_depot = float('inf')
    for i, cust in enumerate(available_customers):
        travel = np.linalg.norm(current_position - cust)
        depot_dist = np.linalg.norm(cust - depot_position)
        new_active_return = travel + depot_dist
        new_max = max(others_max, new_active_return)
        # select based on new_max, then travel, then depot_dist
        if (new_max < best_max or
            (np.isclose(new_max, best_max) and travel < best_travel) or
            (np.isclose(new_max, best_max) and np.isclose(travel, best_travel) and depot_dist < best_depot)):
            best_max = new_max
            best_travel = travel
            best_depot = depot_dist
            best_idx = i
    return best_idx