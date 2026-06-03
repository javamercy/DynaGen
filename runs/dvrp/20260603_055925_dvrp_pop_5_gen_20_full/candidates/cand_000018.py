import numpy as np

def choose_next_customer(current_position: np.ndarray, depot_position: np.ndarray, truck_positions: np.ndarray, available_customers: np.ndarray) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    all_dist = np.linalg.norm(available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2)
    match = np.all(np.isclose(truck_positions, current_position), axis=1)
    distances_to_trucks = all_dist.copy()
    distances_to_trucks[:, match] = np.inf
    d_other_min = np.min(distances_to_trucks, axis=1)
    score = curr_to_cust + 2 * cust_to_depot - d_other_min
    best_idx = np.argmin(score)
    return int(best_idx)