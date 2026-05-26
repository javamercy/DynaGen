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

    # Identify active truck index (closest to current_position)
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))

    # Current distances from each truck to depot
    dist_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)

    # Current statistics for wait comparison and adaptive gamma
    current_max = np.max(dist_to_depot)
    current_min = np.min(dist_to_depot)
    current_mean = np.mean(dist_to_depot)
    current_spread = current_max - current_min
    current_std = np.std(dist_to_depot)

    # Adaptive gamma: scales with spread/mean ratio
    ratio = current_spread / (current_mean + 1e-8)
    gamma = 0.2 * np.clip(ratio, 0.0, 5.0)
    beta = 0.5
    delta = 0.1
    increase_penalty_weight = 1.0

    # Score for waiting (no action)
    wait_score = current_max + beta * current_mean + gamma * current_spread + delta * current_std

    best_idx = None
    best_score = np.inf

    for i, cust in enumerate(available_customers):
        # Active truck's return time after serving this customer
        active_return = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)

        # New return times for all trucks
        new_returns = dist_to_depot.copy()
        new_returns[active_idx] = active_return

        new_max = np.max(new_returns)
        new_min = np.min(new_returns)
        new_mean = np.mean(new_returns)
        new_spread = new_max - new_min
        new_std = np.std(new_returns)

        increase_penalty = increase_penalty_weight * max(0, new_max - current_max)

        score = new_max + beta * new_mean + gamma * new_spread + delta * new_std + increase_penalty

        if score < best_score:
            best_score = score
            best_idx = i

    # Decide whether to wait
    if wait_score < best_score:
        return None
    else:
        return best_idx