import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # compute max distance to depot among other trucks
    other_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dists.append(np.linalg.norm(depot_position - pos))
    max_other = max(other_dists) if other_dists else 0.0
    best_idx = None
    best_score = -float('inf')
    for i, cust in enumerate(available_customers):
        travel = np.linalg.norm(current_position - cust)
        return_dist = np.linalg.norm(cust - depot_position)
        finish = travel + return_dist
        new_max = max(finish, max_other)
        # compute isolation: distance to nearest other truck
        min_ot = float('inf')
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d < min_ot:
                    min_ot = d
        if min_ot == float('inf'):
            min_ot = 0.0
        # score: lower new_max is better, with tiny tie-breaker for isolation
        score = -new_max + 1e-6 * min_ot
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx