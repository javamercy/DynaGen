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
    # Find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    gamma = 0.4
    delta = 0.2

    active_travel = np.linalg.norm(current_position - available_customers, axis=1)
    active_eff = active_travel + (1 - gamma) * depot_dists

    # Single truck case: just minimize travel+depot
    if n_trucks == 1:
        best_idx = np.argmin(active_eff)
        return int(best_idx)

    # Compute minimum effective cost among other trucks
    other_eff = np.full(len(available_customers), np.inf)
    for j in range(n_trucks):
        if j == active_idx:
            continue
        truck_travel = np.linalg.norm(truck_positions[j] - available_customers, axis=1)
        truck_eff = truck_travel + (1 - gamma) * depot_dists
        other_eff = np.minimum(other_eff, truck_eff)

    mask = active_eff <= (1 + delta) * other_eff
    if not np.any(mask):
        return None

    regret = other_eff - active_eff
    # Select best among masked customers: highest regret, tie-break by smallest active_eff
    best_idx = None
    best_regret = -np.inf
    best_active = np.inf
    for i in range(len(available_customers)):
        if mask[i]:
            if regret[i] > best_regret or (regret[i] == best_regret and active_eff[i] < best_active):
                best_regret = regret[i]
                best_active = active_eff[i]
                best_idx = i
    return best_idx