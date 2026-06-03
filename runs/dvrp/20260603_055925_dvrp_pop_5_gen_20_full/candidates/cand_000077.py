import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if available_customers.shape[0] == 0:
        return None

    n_trucks = truck_positions.shape[0]
    if n_trucks == 1:
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))

    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_mask = ~np.all(np.isclose(truck_positions, current_position), axis=1)
    other_positions = truck_positions[other_mask]
    other_depot_dists = np.linalg.norm(other_positions - depot_position, axis=1)
    max_other_depot = np.max(other_depot_dists)

    best_score = -np.inf
    best_idx = None

    for i, cust in enumerate(available_customers):
        cust_depot_dist = np.linalg.norm(cust - depot_position)
        cur_to_cust = np.linalg.norm(cust - current_position)
        final_dist = cur_to_cust + cust_depot_dist
        cur_detour = final_dist - current_depot_dist

        other_to_cust = np.linalg.norm(cust - other_positions, axis=1)
        other_detours = other_to_cust + cust_depot_dist - other_depot_dists
        best_other_detour = np.min(other_detours)

        regret = cur_detour - best_other_detour
        penalty = max(0, final_dist - max_other_depot)
        score = regret - penalty

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return 0
    return int(best_idx)