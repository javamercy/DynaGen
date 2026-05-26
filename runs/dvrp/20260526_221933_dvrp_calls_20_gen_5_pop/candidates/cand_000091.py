import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    c = current_position
    d = depot_position
    dist_to_cust = np.linalg.norm(available_customers - c, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - d, axis=1)
    base = dist_to_cust + cust_to_depot
    # Find index of current truck
    truck_dist = np.linalg.norm(truck_positions - c, axis=1)
    current_truck_idx = int(np.argmin(truck_dist))
    # Distance from each customer to each truck
    cust_to_trucks = np.linalg.norm(available_customers[:, None, :] - truck_positions[None, :, :], axis=2)
    # Exclude current truck
    cust_to_trucks[:, current_truck_idx] = np.inf
    nearest_other_dist = np.min(cust_to_trucks, axis=1)
    # Competition penalty: subtract distance to nearest other truck (so customers far from others get lower score)
    lambda_ = 0.5
    score = base - lambda_ * nearest_other_dist
    # Depot-return urgency: increase weight on cust_to_depot over time
    time_factor = 1 + 0.01 * current_time
    score = score + (time_factor - 1) * cust_to_depot
    # Tie-breaking: prefer larger cust_to_depot
    score = score - 1e-6 * cust_to_depot
    best_idx = int(np.argmin(score))
    return best_idx