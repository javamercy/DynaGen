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

    # Distances
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    active_travel = np.linalg.norm(current_position - available_customers, axis=1)

    # Constants
    gamma = 0.4
    lambd = 0.1  # weight for imbalance penalty

    # Active effective cost (base)
    active_eff = active_travel + (1 - gamma) * depot_dists

    # Single truck case
    if n_trucks == 1:
        best_idx = np.argmin(active_eff)
        return int(best_idx)

    # Compute other trucks' effective cost and identify best other truck per customer
    other_truck_eff = np.full((n_trucks, len(available_customers)), np.inf)
    for j in range(n_trucks):
        if j == active_idx:
            continue
        travel_j = np.linalg.norm(truck_positions[j] - available_customers, axis=1)
        other_truck_eff[j] = travel_j + (1 - gamma) * depot_dists

    other_eff = np.min(other_truck_eff, axis=0)
    best_other_truck_idx = np.argmin(other_truck_eff, axis=0)  # index of truck achieving min

    # Compute depot distance of best other truck for each customer
    # best_other_truck_idx gives which truck (0 to n_trucks-1), but we need its depot distance
    other_truck_depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)  # (n_trucks,)
    # For each customer, get the depot distance of the best other truck
    # best_other_truck_idx might be same as active_idx for customers where no other? but we only consider other trucks, so min over j!=active, so it's fine
    best_other_depot_dist = other_truck_depot_dists[best_other_truck_idx]

    # Imbalance penalty: penalize active_eff when best other truck is close to depot
    epsilon = 1e-6
    penalty = lambd * (1.0 / (best_other_depot_dist + epsilon))
    active_eff_penalized = active_eff + penalty

    # Dynamic slack based on active truck's depot distance relative to average
    active_depot_dist = np.linalg.norm(current_position - depot_position)
    avg_depot_dist = np.mean(np.linalg.norm(truck_positions - depot_position, axis=1))
    ratio = active_depot_dist / (avg_depot_dist + epsilon)
    delta = min(0.5, max(0.1, ratio * 0.3))

    # Apply slack condition: active_eff_penalized <= (1+delta) * other_eff
    mask = active_eff_penalized <= (1 + delta) * other_eff
    if not np.any(mask):
        return None

    # Compute regret as difference in effective costs (using penalized active)
    regret = other_eff - active_eff_penalized
    # Among masked, select highest regret, tie-break by smallest active_eff_penalized
    best_idx = None
    best_regret = -np.inf
    best_active = np.inf
    for i in range(len(available_customers)):
        if mask[i]:
            if regret[i] > best_regret or (regret[i] == best_regret and active_eff_penalized[i] < best_active):
                best_regret = regret[i]
                best_active = active_eff_penalized[i]
                best_idx = i
    return best_idx