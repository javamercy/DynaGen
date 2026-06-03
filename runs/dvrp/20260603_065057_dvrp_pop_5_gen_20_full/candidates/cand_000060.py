import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    n_trucks = len(truck_positions)
    best_saving = -np.inf
    best_idx = None

    for i, cust in enumerate(available_customers):
        dist_cur = np.linalg.norm(current_position - cust)
        min_other = np.inf
        for j, pos in enumerate(truck_positions):
            if np.array_equal(pos, current_position):
                continue
            d = np.linalg.norm(pos - cust)
            if d < min_other:
                min_other = d
        saving = min_other - dist_cur
        if saving > best_saving:
            best_saving = saving
            best_idx = i

    if best_saving <= 0:
        return None
    else:
        return best_idx