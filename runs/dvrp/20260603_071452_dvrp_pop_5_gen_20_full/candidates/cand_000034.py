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
    # Find index of active truck
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    # Compute depot distances
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)

    # Parameters for depot-return bonus and slack
    gamma = 0.15  # depot-return bonus factor
    delta = 0.08  # slack threshold

    # Modified cost: reduce depot distance component
    def effective_cost(start, customer_depot_dist):
        travel = np.linalg.norm(start - customer[None,:], axis=1) if start.ndim == 1 else np.linalg.norm(start[:, None] - customer[None,:], axis=2)
        return travel + (1 - gamma) * customer_depot_dist

    # Compute effective costs for active truck
    active_travel = np.linalg.norm(current_position - available_customers, axis=1)
    active_eff = active_travel + (1 - gamma) * depot_dists

    # Compute minimum effective cost among other trucks
    other_eff = np.full(len(available_customers), np.inf)
    for j in range(n_trucks):
        if j == active_idx:
            continue
        truck_travel = np.linalg.norm(truck_positions[j] - available_customers, axis=1)
        truck_eff = truck_travel + (1 - gamma) * depot_dists
        other_eff = np.minimum(other_eff, truck_eff)

    # If only one truck, pick the customer minimizing actual travel + depot (unmodified) to be consistent
    if n_trucks == 1:
        best_idx = np.argmin(active_travel + depot_dists)
        return int(best_idx)

    # Determine candidates where active cost <= (1+delta)*other cost
    mask = active_eff <= (1 + delta) * other_eff
    if not np.any(mask):
        return None

    # Among candidates, pick the one with largest regret (other_eff - active_eff), tie-break by smallest active_eff
    regret = other_eff - active_eff
    # Only consider masked
    best_idx = None
    best_regret = -np.inf
    best_active = np.inf
    for i in range(len(available_customers)):
        if mask[i]:
            if regret[i] > best_regret or (regret[i] == best_regret and active_eff[i] < best_active):
                best_regret = regret[i]
                best_active = active_eff[i]
                best_idx = i
    return int(best_idx)