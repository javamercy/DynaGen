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
    my_depot_dist = np.linalg.norm(c - d)
    avg_depot_dist = np.mean(np.linalg.norm(truck_positions - d, axis=1))
    depot_penalty = 0.2 if my_depot_dist > avg_depot_dist else 0.0
    tie_break = 1e-6
    score = dist_to_cust + cust_to_depot + depot_penalty * cust_to_depot - tie_break * cust_to_depot
    best_idx = int(np.argmin(score))
    return best_idx