import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    n_trucks = len(truck_positions)
    # find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # distances from each truck to depot (constant for other trucks)
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max_other = 0.0
    for j in range(n_trucks):
        if j != active_idx:
            if depot_dists[j] > current_max_other:
                current_max_other = depot_dists[j]

    # precompute distances from active to customers, and customers to depot
    dist_active_to_cust = np.linalg.norm(current_position - available_customers, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)

    best_idx = None
    best_potential = float('inf')
    best_active_time = float('inf')

    for i in range(len(available_customers)):
        active_time = dist_active_to_cust[i] + dist_cust_to_depot[i]
        potential = max(active_time, current_max_other)
        # only serve if potential does not exceed current_max_other by a significant margin?
        # here we require that the potential TTT is not larger than current_max_other
        # (i.e., serving does not increase the max return time).
        if potential > current_max_other:
            continue
        if potential < best_potential or (potential == best_potential and active_time < best_active_time):
            best_potential = potential
            best_active_time = active_time
            best_idx = i

    return best_idx