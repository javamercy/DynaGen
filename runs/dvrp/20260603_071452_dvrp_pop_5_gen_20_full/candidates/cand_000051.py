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
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)

    # Single truck: simple greedy
    if n_trucks == 1:
        active_travel = np.linalg.norm(available_customers - current_position, axis=1)
        best_idx = np.argmin(active_travel + depot_dists)
        return int(best_idx)

    # Multi-truck: angular clustering
    K = 4  # quadrants
    eps = 1e-9
    # Compute angles for customers and all trucks
    cust_angles = np.arctan2(available_customers[:, 1] - depot_position[1],
                             available_customers[:, 0] - depot_position[0])
    truck_angles = np.arctan2(truck_positions[:, 1] - depot_position[1],
                              truck_positions[:, 0] - depot_position[0])

    # Define bins: from -pi to pi with K equal intervals
    bin_edges = np.linspace(-np.pi, np.pi, K+1)
    # Digitize: bin index 0 to K-1, handling -pi edge (shift by eps to include -pi in bin 0)
    bin_ids_cust = np.digitize(cust_angles, bin_edges, right=True) - 1
    bin_ids_truck = np.digitize(truck_angles, bin_edges, right=True) - 1

    # Count customers and trucks per bin
    cust_counts = np.zeros(K, dtype=int)
    truck_counts = np.zeros(K, dtype=int)
    for b in bin_ids_cust:
        cust_counts[b] += 1
    for b in bin_ids_truck:
        truck_counts[b] += 1

    # Compute load = customers / (trucks + 1)
    loads = cust_counts / (truck_counts + 1)

    # Find bin with max load
    max_load = np.max(loads)
    best_bin = np.argmax(loads)

    # Wait if max load < 1.0 (more trucks than customers in that bin)
    if max_load < 1.0:
        return None

    # Customers in best bin
    in_bin_mask = bin_ids_cust == best_bin
    if not np.any(in_bin_mask):
        return None

    # Among those, select one with minimum active_cost = travel + depot distance
    active_travel = np.linalg.norm(available_customers[in_bin_mask] - current_position, axis=1)
    active_costs = active_travel + depot_dists[in_bin_mask]
    best_in_bin_idx = np.argmin(active_costs)

    # Convert to original index
    original_indices = np.where(in_bin_mask)[0]
    return int(original_indices[best_in_bin_idx])