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
    total_dist = dist_to_cust + cust_to_depot
    score = total_dist - 1e-6 * cust_to_depot
    best_idx = int(np.argmin(score))
    return best_idx