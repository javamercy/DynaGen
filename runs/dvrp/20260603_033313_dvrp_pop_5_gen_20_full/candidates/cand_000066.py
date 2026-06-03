import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    ratio = dist_depot / (dist_current + 1e-8)
    best_idx = np.argmax(ratio)
    if ratio[best_idx] < 0.5:
        return None
    return int(best_idx)