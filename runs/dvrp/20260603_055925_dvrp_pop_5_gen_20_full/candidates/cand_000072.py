import numpy as np

def choose_next_customer(current_position: np.ndarray, depot_position: np.ndarray, truck_positions: np.ndarray, available_customers: np.ndarray) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # distances from current truck to each customer
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # distances from each customer to every truck
    all_dist = np.linalg.norm(available_customers[:, np.newaxis, :] - truck_positions[np.newaxis, :, :], axis=2)
    # identify which truck is the current truck (by position matching)
    current_mask = np.all(np.isclose(truck_positions, current_position), axis=1)
    # set distances to current truck to infinity so it's excluded from min
    all_dist[:, current_mask] = np.inf
    # minimum distance to other trucks for each customer
    min_other = np.min(all_dist, axis=1)
    # composite score (lower is better)
    score = curr_to_cust + cust_to_depot - min_other
    best_idx = int(np.argmin(score))
    # waiting condition: if best customer's round trip > max distance from other trucks to depot, wait
    other_mask = ~current_mask
    if np.any(other_mask):
        other_to_depot = np.linalg.norm(truck_positions[other_mask] - depot_position, axis=1)
        max_other_to_depot = np.max(other_to_depot)
        best_total = curr_to_cust[best_idx] + cust_to_depot[best_idx]
        if best_total > max_other_to_depot:
            return None
    return best_idx