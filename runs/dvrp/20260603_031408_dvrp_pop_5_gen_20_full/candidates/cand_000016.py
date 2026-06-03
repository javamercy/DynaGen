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
        cost = dist_to_truck - 0.5 * dist_to_depot - 0.3 * min_other_dist
        if cost < best_cost:
            best_cost = cost
            best_idx = i
    if best_idx is not None and len(truck_positions) > 1:
        best_cust = available_customers[best_idx]
        d_truck_depot = np.linalg.norm(current_position - depot_position)
        d_truck_cust = np.linalg.norm(current_position - best_cust)
        d_cust_depot = np.linalg.norm(depot_position - best_cust)
        dists_to_best = np.linalg.norm(truck_positions - best_cust, axis=1)
        sorted_dists_best = np.sort(dists_to_best)
        if len(sorted_dists_best) >= 2:
            min_other_best = sorted_dists_best[1]
        else:
            min_other_best = sorted_dists_best[0]
        if d_truck_depot < 0.5 * d_cust_depot and d_truck_cust > 2 * d_truck_depot and min_other_best > d_truck_depot:
            return None
    return best_idx