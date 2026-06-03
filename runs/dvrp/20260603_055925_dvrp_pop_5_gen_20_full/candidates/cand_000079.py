import numpy as np

def choose_next_customer(current_position: np.ndarray, depot_position: np.ndarray, truck_positions: np.ndarray, available_customers: np.ndarray) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_avail = available_customers.shape[0]
    alpha = 2.0
    weight = 1.0 + alpha / (n_avail + 1.0)
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # distances from each customer to every truck
    all_dist = np.linalg.norm(available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2)
    # identify the current truck's index (assuming unique match)
    match = np.all(np.isclose(truck_positions, current_position), axis=1)
    # set distance to current truck to inf
    distances_to_trucks = all_dist.copy()
    distances_to_trucks[:, match] = np.inf
    min_other = np.min(distances_to_trucks, axis=1)
    score = curr_to_cust + weight * cust_to_depot - min_other
    best_idx = np.argmin(score)
    return int(best_idx)