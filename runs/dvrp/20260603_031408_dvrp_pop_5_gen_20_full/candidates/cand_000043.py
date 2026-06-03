import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # identify active truck index
    mask = np.all(np.isclose(truck_positions, current_position), axis=1)
    active_idx = np.where(mask)[0][0]
    # other trucks distances to depot
    other_dists = np.delete(
        np.linalg.norm(truck_positions - depot_position, axis=1),
        active_idx
    )
    max_other_dist = np.max(other_dists) if len(other_dists) > 0 else 0.0
    best_idx = None
    best_makespan = float('inf')
    best_total = float('inf')
    best_travel = float('inf')
    for i, cust in enumerate(available_customers):
        travel = np.linalg.norm(current_position - cust)
        dist_to_depot = np.linalg.norm(cust - depot_position)
        active_total = travel + dist_to_depot
        makespan = max(active_total, max_other_dist)
        # lexicographic tie-breaking
        if (makespan < best_makespan or
            (np.isclose(makespan, best_makespan) and active_total < best_total) or
            (np.isclose(makespan, best_makespan) and np.isclose(active_total, best_total) and travel < best_travel)):
            best_makespan = makespan
            best_total = active_total
            best_travel = travel
            best_idx = i
    return best_idx