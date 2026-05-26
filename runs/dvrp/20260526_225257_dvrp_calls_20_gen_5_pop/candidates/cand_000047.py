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

    # Parameters
    beta0 = 1.0
    gamma0 = 0.5
    lambda_depot = 0.1
    decay = 0.02

    # Decay coefficients
    beta = beta0 * np.exp(-decay * current_time)
    gamma = gamma0 * np.exp(-decay * current_time)

    # Current truck's distance to depot
    current_depot_dist = np.linalg.norm(current_position - depot_position)

    # Distances for current truck to customers and to depot
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = curr_to_cust + cust_to_depot

    # Identify other trucks (excluding current truck)
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]

    # Centroid of all trucks for fleet balance
    centroid = np.mean(truck_positions, axis=0)

    # Initialize adjusted costs
    adjusted_costs = cost_now.copy()

    # Compute terms if other trucks exist
    if len(other_trucks) > 0:
        # Isolation penalty: min distance from customer to any other truck
        # Broadcasting: (n_cust, 1, 2) - (1, n_other, 2) -> (n_cust, n_other, 2)
        diff = available_customers[:, None, :] - other_trucks[None, :, :]
        dist_to_other = np.linalg.norm(diff, axis=2)  # (n_cust, n_other)
        iso_pen = np.min(dist_to_other, axis=1)  # (n_cust,)
        adjusted_costs += beta * iso_pen

    # Fleet balance term: distance from customer to centroid
    fleet_bal = np.linalg.norm(available_customers - centroid, axis=1)
    adjusted_costs += gamma * fleet_bal

    # Depot pressure term: increase with time and distance to depot
    adjusted_costs += lambda_depot * current_time * cust_to_depot

    # Tie-breaking: prefer smaller customer-to-depot distance when costs are close
    # Find minimum adjusted cost
    min_cost = np.min(adjusted_costs)
    # Indices with cost within a small epsilon
    eps = 1e-6
    candidates = np.where(adjusted_costs - min_cost < eps)[0]
    if len(candidates) == 1:
        best_idx = candidates[0]
    else:
        # Among candidates, pick the one with smallest depot distance
        best_idx = candidates[np.argmin(cust_to_depot[candidates])]

    # Waiting condition: compare best adjusted cost to threshold based on current depot distance
    threshold = 0.8 * current_depot_dist
    if adjusted_costs[best_idx] < threshold:
        return int(best_idx)
    else:
        return None