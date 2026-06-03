import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # current distances to depot for all trucks
    all_dists_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(all_dists_to_depot)
    # identify other trucks (exclude the one at current_position)
    mask = np.all(np.isclose(truck_positions, current_position), axis=1)
    active_idx = np.where(mask)[0]
    if len(active_idx) > 0:
        other_dists = np.delete(all_dists_to_depot, active_idx[0])
    else:
        other_dists = all_dists_to_depot  # fallback
    others_max = np.max(other_dists) if len(other_dists) > 0 else 0.0
    # iterate over available customers
    best_idx = None
    best_cost = float('inf')
    best_new_active_total = float('inf')
    best_travel = float('inf')
    for i, cust in enumerate(available_customers):
        travel = np.linalg.norm(current_position - cust)
        dep_dist = np.linalg.norm(cust - depot_position)
        new_active_total = travel + dep_dist
        new_max = max(others_max, new_active_total)
        cost = travel + 0.5 * (new_max - current_max)
        # tie-breaking: prefer smaller new_active_total, then smaller travel
        if cost < best_cost:
            best_cost = cost
            best_new_active_total = new_active_total
            best_travel = travel
            best_idx = i
        elif np.isclose(cost, best_cost):
            if new_active_total < best_new_active_total:
                best_cost = cost
                best_new_active_total = new_active_total
                best_travel = travel
                best_idx = i
            elif np.isclose(new_active_total, best_new_active_total) and travel < best_travel:
                best_cost = cost
                best_new_active_total = new_active_total
                best_travel = travel
                best_idx = i
    return best_idx