import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n = len(truck_positions)
    current_dist_to_depot = dist(current_position, depot_position)
    other_dists_to_depot = [dist(truck_positions[j], depot_position) for j in range(n) if not np.array_equal(truck_positions[j], current_position)]
    other_max = max(other_dists_to_depot) if other_dists_to_depot else 0.0

    best_idx = None
    best_max = float('inf')
    best_new_ret = float('inf')
    best_cust_depot = float('inf')

    for i, cust in enumerate(available_customers):
        new_ret = dist(current_position, cust) + dist(cust, depot_position)
        candidate_max = max(new_ret, other_max)
        cust_depot = dist(cust, depot_position)
        if (candidate_max < best_max) or (candidate_max == best_max and new_ret < best_new_ret) or (candidate_max == best_max and new_ret == best_new_ret and cust_depot < best_cust_depot):
            best_max = candidate_max
            best_new_ret = new_ret
            best_cust_depot = cust_depot
            best_idx = i

    wait_max = max(current_dist_to_depot, other_max)
    if best_max <= wait_max:
        return best_idx
    else:
        return None