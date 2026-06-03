import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    # distances from active truck to each customer
    d_active = np.linalg.norm(available_customers - current_position, axis=1)
    # identify active truck index
    diff = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(diff)
    other_indices = [i for i in range(truck_positions.shape[0]) if i != active_idx]
    if len(other_indices) == 0:
        return int(np.argmin(d_active))
    other_trucks = truck_positions[other_indices]
    # min distance from any other truck to each customer
    d_other_min = np.min(np.linalg.norm(available_customers[:, None] - other_trucks[None], axis=2), axis=1)
    # territorial advantage
    advantage = np.where(d_active > 1e-9, d_other_min / d_active, np.inf)
    # depot distances
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    max_depot_dist = np.max(np.linalg.norm(truck_positions - depot_position, axis=1))
    depot_bias = -customer_to_depot / (max_depot_dist + 1e-9)
    # adaptive weight based on active truck's depot distance
    active_depot_dist = np.linalg.norm(current_position - depot_position)
    w = active_depot_dist / (max_depot_dist + 1e-9)
    score = advantage + w * depot_bias
    best = np.argmax(score)
    return int(best)