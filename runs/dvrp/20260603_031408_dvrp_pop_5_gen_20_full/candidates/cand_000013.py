import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_idx = None
    best_cost = float('inf')
    for i, cust in enumerate(available_customers):
        dist_to_truck = np.linalg.norm(current_position - cust)
        dist_to_depot = np.linalg.norm(depot_position - cust)
        dists_to_trucks = np.linalg.norm(truck_positions - cust, axis=1)
        sorted_dists = np.sort(dists_to_trucks)
        if len(sorted_dists) >= 2:
            min_other_dist = sorted_dists[1]
        else:
            min_other_dist = sorted_dists[0]
        cost = dist_to_truck + 0.8 * dist_to_depot - 0.1 * min_other_dist
        if cost < best_cost:
            best_cost = cost
            best_idx = i
    return best_idx