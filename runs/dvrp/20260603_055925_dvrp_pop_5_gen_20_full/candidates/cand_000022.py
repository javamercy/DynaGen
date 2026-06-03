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
    if n_trucks == 1:
        dists = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(dists))
    # Identify current truck index
    current_truck_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # Current max return among all trucks (direct to depot)
    current_max_return = np.max(np.linalg.norm(truck_positions - depot_position, axis=1))
    best_regret = -float('inf')
    best_idx = None
    best_second = -float('inf')
    for i in range(len(available_customers)):
        pot_return_curr = curr_to_cust[i] + cust_to_depot[i]
        # Compute min other return
        other_returns = []
        for j in range(n_trucks):
            if j == current_truck_idx:
                continue
            other_to_cust = np.linalg.norm(truck_positions[j] - available_customers[i])
            other_returns.append(other_to_cust + cust_to_depot[i])
        min_other = min(other_returns) if other_returns else float('inf')
        urgency = max(0.0, pot_return_curr - current_max_return)
        regret = (min_other - pot_return_curr) - urgency
        secondary = -pot_return_curr
        if regret > best_regret or (regret == best_regret and secondary > best_second):
            best_regret = regret
            best_idx = i
            best_second = secondary
    if best_regret >= 0:
        return best_idx
    else:
        return None