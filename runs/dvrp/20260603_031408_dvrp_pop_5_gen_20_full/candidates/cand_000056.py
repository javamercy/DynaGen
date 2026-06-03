import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    other_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dists.append(np.linalg.norm(pos - depot_position))
    max_other_dist = max(other_dists) if other_dists else 0.0
    best_idx = None
    best_makespan = float('inf')
    best_travel = float('inf')
    for i, cust in enumerate(available_customers):
        travel = np.linalg.norm(current_position - cust)
        ret = np.linalg.norm(cust - depot_position)
        completion = travel + ret
        makespan = max(completion, max_other_dist)
        if makespan < best_makespan or (makespan == best_makespan and travel < best_travel):
            best_makespan = makespan
            best_travel = travel
            best_idx = i
    return best_idx