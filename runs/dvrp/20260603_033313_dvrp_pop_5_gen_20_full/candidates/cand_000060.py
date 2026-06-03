import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    idx_self = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    n_trucks = truck_positions.shape[0]
    if n_trucks == 1:
        dist_self = np.linalg.norm(available_customers - current_position, axis=1)
        dist_depot = np.linalg.norm(available_customers - depot_position, axis=1)
        scores = dist_self + dist_depot
        return int(np.argmin(scores))
    else:
        best_idx = -1
        best_score = -np.inf
        best_d_self = np.inf
        for i, cust in enumerate(available_customers):
            dists = np.linalg.norm(truck_positions - cust, axis=1)
            dists[idx_self] = np.inf
            min_other = np.min(dists)
            d_depot = np.linalg.norm(depot_position - cust)
            score = min_other - d_depot
            d_self = np.linalg.norm(current_position - cust)
            if score > best_score or (score == best_score and d_self < best_d_self):
                best_score = score
                best_d_self = d_self
                best_idx = i
        return best_idx