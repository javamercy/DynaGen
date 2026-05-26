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
    # Competition penalty: penalize customers close to other trucks
    if len(truck_positions) > 1:
        dist_to_all_trucks = np.linalg.norm(
            available_customers[:, None, :] - truck_positions[None, :, :],
            axis=2
        )
        sorted_dists = np.sort(dist_to_all_trucks, axis=1)
        min_other_dist = sorted_dists[:, 1]  # second smallest (closest other truck)
        competition_penalty = 1.0 / (min_other_dist + 1e-6)
    else:
        competition_penalty = 0.0
    # Depot-return urgency weight increases with time
    time_weight = 0.001
    depot_weight = 1 + time_weight * current_time
    # Score: minimize, with tie-break favoring farther customers
    score = dist_to_cust + depot_weight * cust_to_depot + competition_penalty - 1e-6 * cust_to_depot
    best_idx = int(np.argmin(score))
    return best_idx