import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # compute current max distance from depot among other trucks
    other_dists = []
    for pos in truck_positions:
        if not np.array_equal(pos, current_position):
            other_dists.append(np.linalg.norm(depot_position - pos))
    max_other = max(other_dists) if other_dists else 0.0
    best_idx = None
    best_score = float('inf')
    best_iso = -float('inf')
    for i, cust in enumerate(available_customers):
        finish = np.linalg.norm(current_position - cust) + np.linalg.norm(depot_position - cust)
        candidate_max = max(finish, max_other)
        # isolation: distance to nearest other truck
        min_ot = float('inf')
        for pos in truck_positions:
            if not np.array_equal(pos, current_position):
                d = np.linalg.norm(pos - cust)
                if d < min_ot:
                    min_ot = d
        if min_ot == float('inf'):
            min_ot = 0.0
        # tie-break: prefer larger isolation
        if candidate_max < best_score or (candidate_max == best_score and min_ot > best_iso):
            best_score = candidate_max
            best_iso = min_ot
            best_idx = i
    return best_idx