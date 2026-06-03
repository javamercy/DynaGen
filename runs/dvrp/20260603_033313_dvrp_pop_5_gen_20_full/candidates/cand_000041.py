import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Compute distances to depot for all trucks
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    # Identify current truck index
    current_idx = None
    for i, pos in enumerate(truck_positions):
        if np.allclose(pos, current_position):
            current_idx = i
            break
    if current_idx is None:
        # Should not happen, but fallback: assume first truck?
        current_idx = 0
    # Other trucks' distances to depot
    other_dist = np.delete(dist_to_depot, current_idx)
    other_max = other_dist.max() if other_dist.size > 0 else 0.0
    cur_to_depot = dist_to_depot[current_idx]
    # Evaluate waiting
    makespan_wait = max(cur_to_depot, other_max)
    # Evaluate each customer
    best_idx = None
    best_makespan = float('inf')
    best_cost_now = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        makespan_serve = max(cost_now, other_max)
        if makespan_serve < best_makespan or (makespan_serve == best_makespan and cost_now < best_cost_now):
            best_makespan = makespan_serve
            best_cost_now = cost_now
            best_idx = i
    # Compare with waiting
    if makespan_wait <= best_makespan:
        return None
    else:
        return best_idx