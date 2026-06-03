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
    # find active truck index
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        raise ValueError("current_position not found in truck_positions")

    n_cust = len(available_customers)
    # depot distances
    depot_dist = np.linalg.norm(available_customers - depot_position, axis=1)
    # active costs
    active_costs = np.linalg.norm(current_position - available_customers, axis=1) + depot_dist

    # if only one truck, go to nearest customer (including depot return)
    if n_trucks == 1:
        return int(np.argmin(active_costs))

    # For each other truck, compute best and second best costs over all customers
    other_best = np.full(n_trucks, np.inf)  # best cost for each truck
    other_best_idx = np.full(n_trucks, -1, dtype=int)  # index of best customer
    other_second_best = np.full(n_trucks, np.inf)  # second best cost

    for j in range(n_trucks):
        if j == active_idx:
            continue
        costs_j = np.linalg.norm(truck_positions[j] - available_customers, axis=1) + depot_dist
        # get two smallest indices
        sorted_indices = np.argsort(costs_j)
        best_idx_local = sorted_indices[0]
        best_val = costs_j[best_idx_local]
        sec_val = costs_j[sorted_indices[1]]
        other_best[j] = best_val
        other_best_idx[j] = best_idx_local
        other_second_best[j] = sec_val

    # compute max of other trucks' best (baseline when active does nothing)
    no_action_max = np.max(other_best[other_best != np.inf])

    # For each customer, compute the max over other trucks of their best after removing that customer
    other_max_per_cust = np.full(n_cust, -np.inf)
    for j in range(n_trucks):
        if j == active_idx:
            continue
        best_val = other_best[j]
        best_idx = other_best_idx[j]
        sec_val = other_second_best[j]
        # create array with best_val for all customers, then replace at best_idx with sec_val
        vals = np.full(n_cust, best_val)
        vals[best_idx] = sec_val
        other_max_per_cust = np.maximum(other_max_per_cust, vals)

    # overall max if active takes customer i
    overall_max_if_take = np.maximum(active_costs, other_max_per_cust)

    # find best customer
    min_max = np.min(overall_max_if_take)
    candidate_indices = np.where(overall_max_if_take == min_max)[0]
    # among those, choose with smallest active cost
    best_candidate = candidate_indices[np.argmin(active_costs[candidate_indices])]

    if min_max < no_action_max:
        return int(best_candidate)
    else:
        return None