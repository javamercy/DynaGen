import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    mask = np.all(np.isclose(truck_positions, current_position), axis=1)
    active_idx = np.where(mask)[0]
    if len(active_idx) == 0:
        active_idx = np.array([0])
    active_idx = active_idx[0]
    active_depot_dist = depot_dists[active_idx]
    other_depot_dists = np.delete(depot_dists, active_idx)
    current_max = np.max(depot_dists)
    best_idx = None
    best_makespan = float('inf')
    best_travel = float('inf')
    best_new_active = float('inf')
    for i, cust in enumerate(available_customers):
        travel = np.linalg.norm(current_position - cust)
        new_active_depot = travel + np.linalg.norm(cust - depot_position)
        candidate_makespan = new_active_depot
        if len(other_depot_dists) > 0:
            candidate_makespan = max(new_active_depot, np.max(other_depot_dists))
        if candidate_makespan < best_makespan:
            best_makespan = candidate_makespan
            best_travel = travel
            best_new_active = new_active_depot
            best_idx = i
        elif np.isclose(candidate_makespan, best_makespan):
            if travel < best_travel:
                best_makespan = candidate_makespan
                best_travel = travel
                best_new_active = new_active_depot
                best_idx = i
            elif np.isclose(travel, best_travel) and new_active_depot < best_new_active:
                best_makespan = candidate_makespan
                best_travel = travel
                best_new_active = new_active_depot
                best_idx = i
    return best_idx