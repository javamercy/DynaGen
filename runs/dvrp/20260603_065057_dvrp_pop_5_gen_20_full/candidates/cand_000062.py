import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    # Identify current truck index
    truck_idx = None
    for i, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            truck_idx = i
            break
    if truck_idx is None:
        # Fallback: nearest truck
        dists = np.linalg.norm(truck_positions - current_position, axis=1)
        truck_idx = int(np.argmin(dists))

    n_trucks = len(truck_positions)
    n_cust = len(available_customers)

    # Precompute distances
    dists_truck_to_cust = np.linalg.norm(
        available_customers[np.newaxis, :, :] - truck_positions[:, np.newaxis, :], axis=2
    )  # shape (n_trucks, n_cust)
    dists_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)  # (n_cust,)
    dists_truck_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)  # (n_trucks,)

    alpha = 0.5  # penalty factor for truck distance to depot

    # Current truck costs
    current_costs = (
        dists_truck_to_cust[truck_idx]  # from truck to customer
        + dists_cust_to_depot  # from customer to depot
        + alpha * dists_truck_to_depot[truck_idx]  # penalty for current truck being far
    )

    best_regret = -np.inf
    best_idx = None

    for c in range(n_cust):
        # Find minimum cost among other trucks
        other_costs = (
            dists_truck_to_cust[:, c]
            + dists_cust_to_depot[c]
            + alpha * dists_truck_to_depot
        )
        # Exclude current truck
        other_costs[truck_idx] = np.inf
        min_other_cost = np.min(other_costs)

        regret = min_other_cost - current_costs[c]
        if regret > best_regret:
            best_regret = regret
            best_idx = c

    if best_regret > 0 and best_idx is not None:
        return int(best_idx)
    else:
        return None