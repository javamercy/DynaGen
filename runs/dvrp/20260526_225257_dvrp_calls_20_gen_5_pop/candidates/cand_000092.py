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

    # cost for this truck: travel to customer + return to depot
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + customer_to_depot

    # identify other trucks (exclude current position)
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]

    if len(other_trucks) == 0:
        best_idx = np.argmin(cost_now)
        return int(best_idx)

    # best other cost for each customer
    other_to_customer = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )
    best_other_cost = np.min(other_to_customer, axis=1) + customer_to_depot

    regret = best_other_cost - cost_now
    max_regret = np.max(regret)

    if max_regret > 1e-6:
        best_idx = np.argmax(regret)
        return int(best_idx)

    # No positive regret: consider waiting with original condition
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    is_closest_to_depot = current_depot_dist < np.min(other_depot_dists) - 1e-6

    # Wait if closest to depot and available customers <= number of other trucks
    if is_closest_to_depot and len(available_customers) <= len(other_trucks):
        return None
    else:
        # Bias for customers that reduce fleet imbalance: far from other trucks
        # Scale distance to other trucks (max distance) to avoid numerical issues
        other_to_customer = np.linalg.norm(
            available_customers[:, None, :] - other_trucks[None, :, :], axis=2
        )
        max_distance_to_other = np.max(other_to_customer, axis=1) if len(other_trucks) > 0 else 0
        adjusted_cost = cost_now - 0.01 * max_distance_to_other
        best_idx = np.argmin(adjusted_cost)
        return int(best_idx)