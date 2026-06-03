import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    # route distance for each truck to each customer: dist(truck, customer) + dist(customer, depot)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    truck_to_cust = np.linalg.norm(truck_positions[:, np.newaxis, :] - available_customers[np.newaxis, :, :], axis=2)  # (n_trucks, n_customers)
    route_distances = truck_to_cust + cust_to_depot[np.newaxis, :]  # (n_trucks, n_customers)

    # Find the best route distance for each customer (minimum over trucks)
    best_route = np.min(route_distances, axis=0)  # (n_customers,)

    # Determine which customers have the active truck as one of the best (equal to best_route)
    active_idx = np.where(np.all(np.isclose(truck_positions, current_position), axis=1))[0][0]
    active_route = route_distances[active_idx, :]  # (n_customers,)
    is_best = np.isclose(active_route, best_route)  # (n_customers,)

    candidates = np.where(is_best)[0]
    if len(candidates) == 0:
        return None

    # Among candidates, select the one with largest cust_to_depot
    # Tie-break: smallest distance from current_position to customer, then smallest index
    cand_cust_to_depot = cust_to_depot[candidates]
    max_depot = np.max(cand_cust_to_depot)
    depot_candidates = candidates[cand_cust_to_depot == max_depot]

    if len(depot_candidates) == 1:
        return int(depot_candidates[0])

    # Tie-break by smallest current_to_cust distance
    curr_to_cust = np.linalg.norm(current_position - available_customers[depot_candidates], axis=1)
    min_dist = np.min(curr_to_cust)
    dist_candidates = depot_candidates[curr_to_cust == min_dist]

    # Final tie-break: smallest index
    return int(dist_candidates[0])