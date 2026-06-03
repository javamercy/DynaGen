import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # distances to depot for all trucks
    dists_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # identify active truck index (may have multiple at same position? take first)
    active_mask = np.all(np.isclose(truck_positions, current_position), axis=1)
    active_idx = np.where(active_mask)[0][0]
    active_dist = dists_to_depot[active_idx]
    # other trucks' distances
    other_dists = np.delete(dists_to_depot, active_idx)
    max_other = np.max(other_dists) if len(other_dists) > 0 else 0.0
    current_makespan = np.max(dists_to_depot)
    # evaluate each customer
    best_idx = None
    best_new_makespan = float('inf')
    best_travel = float('inf')
    for i, cust in enumerate(available_customers):
        travel = np.linalg.norm(current_position - cust)
        cust_depot_dist = np.linalg.norm(cust - depot_position)
        new_active_total = travel + cust_depot_dist
        new_makespan = max(max_other, new_active_total)
        if new_makespan < best_new_makespan:
            best_new_makespan = new_makespan
            best_travel = travel
            best_idx = i
        elif new_makespan == best_new_makespan and travel < best_travel:
            best_travel = travel
            best_idx = i
    # waiting condition: if active truck is close to depot and other trucks are far, and best travel is high relative to active distance
    if best_idx is not None:
        if active_dist < 0.3 * max_other and best_travel > 2 * active_dist:
            return None
        else:
            return best_idx
    else:
        return None