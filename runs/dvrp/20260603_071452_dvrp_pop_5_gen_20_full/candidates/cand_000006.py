import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    d_active = np.linalg.norm(available_customers - current_position, axis=1)
    # centroid of all available customers
    centroid = np.mean(available_customers, axis=0)
    d_centroid = np.linalg.norm(available_customers - centroid, axis=1)
    d_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # distance to nearest other truck
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    other_indices = [i for i in range(truck_positions.shape[0]) if i != active_idx]
    if len(other_indices) > 0:
        other_trucks = truck_positions[other_indices]
        d_other_other = np.min(np.linalg.norm(available_customers[:, None] - other_trucks[None], axis=2), axis=1)
    else:
        # if only one truck, set d_other to a large value (not used effectively)
        d_other_other = np.ones_like(d_active) * 1e9
    # compute multiplicative score, avoid division by zero
    epsilon = 1e-9
    ratio = d_other_other / (d_active + epsilon)
    score = ratio * (1 / (1 + d_centroid)) * (1 / (1 + d_depot))
    best_idx = int(np.argmax(score))
    # wait condition: if best customer's distance to active > 2 * average distance of customers to depot
    avg_depot_dist = np.mean(d_depot)
    if d_active[best_idx] > 2 * avg_depot_dist:
        return None
    return best_idx