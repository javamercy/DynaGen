import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    best_idx = None
    best_score = -float('inf')
    for i, cust in enumerate(available_customers):
        dist_truck = np.linalg.norm(current_position - cust)
        dist_depot = np.linalg.norm(depot_position - cust)
        score = dist_depot / (dist_truck + 1e-6)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx