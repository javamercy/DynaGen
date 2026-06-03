import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    current_depot_dists = [dist(truck_positions[j], depot_position) for j in range(n_trucks)]
    max_all_current = max(current_depot_dists)
    cur_to_dep = dist(current_position, depot_position)

    best_idx = None
    best_new_return = float('inf')

    for i, cust in enumerate(available_customers):
        new_return = dist(current_position, cust) + dist(cust, depot_position)
        if new_return <= max_all_current:  # serving does not increase the maximum
            if new_return < best_new_return:
                best_new_return = new_return
                best_idx = i

    if best_idx is not None:
        return best_idx
    else:
        return None