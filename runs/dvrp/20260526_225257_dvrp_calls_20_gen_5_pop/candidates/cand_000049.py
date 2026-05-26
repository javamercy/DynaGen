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

    # Basic costs
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + customer_to_depot

    # Identify other trucks
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]

    if len(other_trucks) == 0:
        best_idx = np.argmin(cost_now)
        return int(best_idx)

    # Best other cost for each customer
    other_to_customer = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )
    best_other_cost = np.min(other_to_customer, axis=1) + customer_to_depot
    regret = best_other_cost - cost_now

    # Centroid of available customers
    centroid = available_customers.mean(axis=0)
    dist_to_centroid = np.linalg.norm(available_customers - centroid, axis=1)
    centroid_bonus = -dist_to_centroid  # encourage moving towards centroid

    # Depot reduction bonus
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    depot_reduction = current_depot_dist - customer_to_depot  # positive if customer is closer to depot

    # Combined score with tunable weights
    alpha = 0.1
    beta = 0.05
    score = regret + alpha * centroid_bonus + beta * depot_reduction

    max_regret = np.max(regret)

    if max_regret > 1e-6:
        # Urgent: pick best score
        best_idx = np.argmax(score)
        return int(best_idx)
    else:
        # No positive regret: consider waiting
        other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
        max_other_depot = np.max(other_depot_dists) if len(other_depot_dists) > 0 else 0.0
        is_farthest = current_depot_dist >= max_other_depot - 1e-6

        if not is_farthest and len(available_customers) > 3 and np.min(cost_now) >= current_depot_dist:
            # Wait: other trucks are better positioned and no very cheap customer nearby
            return None
        else:
            best_idx = np.argmax(score)  # highest score, likely min cost_now
            return int(best_idx)