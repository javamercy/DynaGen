import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n = len(available_customers)
    n_trucks = len(truck_positions)
    best_idx = None
    best_score = -np.inf
    cur = current_position
    avail = available_customers
    for i in range(n):
        cust = avail[i]
        d_cur = np.linalg.norm(cur - cust)
        d_depot = np.linalg.norm(cust - depot_position)
        # nearest other customer distance
        if n == 1:
            min_nn = 0.0
        else:
            min_nn = np.inf
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(avail[i] - avail[j])
                    if d < min_nn:
                        min_nn = d
        # distance to nearest other truck
        if n_trucks == 1:
            nearest_other_truck = 0.0
        else:
            dists_to_trucks = np.linalg.norm(truck_positions - cust, axis=1)
            sorted_dists = np.sort(dists_to_trucks)
            # smallest is 0 (self), second smallest is nearest other truck
            nearest_other_truck = sorted_dists[1] if len(sorted_dists) > 1 else 0.0
        score = min_nn - d_cur - 0.5 * d_depot + 0.3 * nearest_other_truck
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx