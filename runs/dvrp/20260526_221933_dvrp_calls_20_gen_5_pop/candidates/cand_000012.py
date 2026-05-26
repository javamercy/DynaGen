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
    # distances from current truck to customers
    dist_to_cust = np.linalg.norm(available_customers - c, axis=1)
    # customers to depot
    cust_to_depot = np.linalg.norm(available_customers - d, axis=1)
    # identify current truck index
    idx_current = np.argmin(np.linalg.norm(truck_positions - c, axis=1))
    # competition penalty: for each customer, minimum distance to any other truck
    min_other_dists = np.empty(len(available_customers))
    for i, cust in enumerate(available_customers):
        dists_to_trucks = np.linalg.norm(truck_positions - cust, axis=1)
        dists_to_trucks[idx_current] = np.inf
        min_other_dists[i] = np.min(dists_to_trucks)
    competition_weight = 0.5
    tie_break = 1e-6
    score = dist_to_cust + cust_to_depot + competition_weight / (min_other_dists + 1e-6) - tie_break * cust_to_depot
    best_idx = int(np.argmin(score))
    return best_idx