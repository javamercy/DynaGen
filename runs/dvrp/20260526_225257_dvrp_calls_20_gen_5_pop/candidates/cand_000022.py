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
    # distances from current to customer and customer to depot
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    own_score = current_to_customer + customer_to_depot
    # distances to nearest other truck (excluding current)
    other_trucks = np.array([p for p in truck_positions if not np.allclose(p, current_position)])
    if len(other_trucks) == 0:
        nearest_other = np.full(len(available_customers), np.inf)
    else:
        dist_to_other = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)
        nearest_other = np.min(dist_to_other, axis=1)
    # adaptive beta based on number of remaining customers
    n_avail = len(available_customers)
    beta = 1.5 * n_avail / (n_avail + 5.0)
    # time-dependent weight on customer_to_depot
    gamma = 0.3
    delta = 0.02
    time_weight = delta * current_time
    effective_gamma = gamma - time_weight  # initially positive, becomes negative as time increases
    score = own_score - beta * nearest_other - effective_gamma * customer_to_depot
    best_idx = np.argmin(score)
    return int(best_idx)