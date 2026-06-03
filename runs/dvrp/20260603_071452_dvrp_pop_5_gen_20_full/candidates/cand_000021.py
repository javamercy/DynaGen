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
    # identify active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    gamma = 0.2  # depot bonus from parent
    n_cust = len(available_customers)

    # Precompute distances to depot and cost matrix
    depot_dists = np.linalg.norm(available_customers - depot_position, axis=1)
    cost = np.zeros((n_trucks, n_cust))
    for t in range(n_trucks):
        travel = np.linalg.norm(truck_positions[t] - available_customers, axis=1)
        cost[t] = travel + (1 - gamma) * depot_dists

    # current direct-depot makespan (baseline)
    direct_to_depot = np.zeros(n_trucks)
    for t in range(n_trucks):
        direct_to_depot[t] = np.linalg.norm(truck_positions[t] - depot_position)
    baseline_makespan = np.max(direct_to_depot)

    best_makespan = np.inf
    best_idx = None

    # For each customer candidate
    for i in range(n_cust):
        active_cost = cost[active_idx, i]
        # remaining customers (all except i)
        remaining = list(range(n_cust))
        remaining.remove(i)
        # compute other trucks' best cost from remaining
        other_best = []
        for j in range(n_trucks):
            if j == active_idx:
                continue
            if len(remaining) > 0:
                best = np.min(cost[j, remaining])
            else:
                best = direct_to_depot[j]  # go directly to depot
            other_best.append(best)
        makespan = max(active_cost, max(other_best))
        if makespan < best_makespan or (makespan == best_makespan and active_cost < best_active_cost):
            best_makespan = makespan
            best_active_cost = active_cost
            best_idx = i

    if best_idx is not None and best_makespan < baseline_makespan:
        return best_idx
    else:
        return None