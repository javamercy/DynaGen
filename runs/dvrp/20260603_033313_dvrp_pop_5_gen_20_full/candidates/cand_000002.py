import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    # Find index of the current truck
    # Since current_position is exactly the position of this truck, we find the closest position
    idx_self = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    n_trucks = truck_positions.shape[0]
    if n_trucks == 1:
        # Only one truck: go to nearest customer
        dists = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(dists))
    else:
        best_idx = -1
        best_min_other = -np.inf
        best_dist_self = np.inf
        for i, cust in enumerate(available_customers):
            dists = np.linalg.norm(truck_positions - cust, axis=1)
            dists[idx_self] = np.inf  # ignore this truck
            min_other = np.min(dists)
            dist_self = np.linalg.norm(current_position - cust)
            # Maximize min_other, then minimize dist_self
            if min_other > best_min_other or (min_other == best_min_other and dist_self < best_dist_self):
                best_min_other = min_other
                best_dist_self = dist_self
                best_idx = i
        return best_idx