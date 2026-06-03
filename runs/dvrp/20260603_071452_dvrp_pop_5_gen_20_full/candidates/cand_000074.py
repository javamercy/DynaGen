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
    active_idx = None
    for i in range(n_trucks):
        if np.allclose(truck_positions[i], current_position):
            active_idx = i
            break
    if active_idx is None:
        active_idx = 0

    depot = depot_position
    truck_depot_dist = np.linalg.norm(truck_positions - depot, axis=1)
    diff = truck_positions[:, np.newaxis, :] - available_customers[np.newaxis, :, :]
    truck_cust_dist = np.linalg.norm(diff, axis=2)
    cust_depot_dist = np.linalg.norm(available_customers - depot, axis=1)
    cost_matrix = truck_cust_dist + cust_depot_dist[np.newaxis, :]

    active_costs = cost_matrix[active_idx]
    other_indices = [j for j in range(n_trucks) if j != active_idx]
    n_others = len(other_indices)

    if n_others == 0:
        return int(np.argmin(active_costs))

    top2_per_j = {}
    direct_cost_j = {}
    best_j_all = {}
    for j in other_indices:
        direct = truck_depot_dist[j]
        direct_cost_j[j] = direct
        cust_costs = cost_matrix[j]
        # find two smallest
        if len(cust_costs) == 0:
            best1_val, best1_idx, best2_val, best2_idx = np.inf, -1, np.inf, -1
        else:
            sorted_indices = np.argsort(cust_costs)
            best1_idx = sorted_indices[0]
            best1_val = cust_costs[best1_idx]
            if len(cust_costs) >= 2:
                best2_idx = sorted_indices[1]
                best2_val = cust_costs[best2_idx]
            else:
                best2_idx = -1
                best2_val = np.inf
        top2_per_j[j] = (best1_val, best1_idx, best2_val, best2_idx)
        best_j_all[j] = min(direct, best1_val)

    baseline_max = max(best_j_all.values())

    best_candidate = None
    best_candidate_max = np.inf
    best_savings = -np.inf

    for i in range(len(available_customers)):
        active_i = active_costs[i]
        max_except_i = 0.0
        for j in other_indices:
            b1_val, b1_idx, b2_val, b2_idx = top2_per_j[j]
            if i == b1_idx:
                best_except = min(direct_cost_j[j], b2_val)
            else:
                best_except = min(direct_cost_j[j], b1_val)
            if best_except > max_except_i:
                max_except_i = best_except
        candidate_max = max(active_i, max_except_i)
        if candidate_max < baseline_max:
            best_other = np.inf
            for j in other_indices:
                if cost_matrix[j][i] < best_other:
                    best_other = cost_matrix[j][i]
            savings = best_other - active_i
            if candidate_max < best_candidate_max or (candidate_max == best_candidate_max and savings > best_savings):
                best_candidate = i
                best_candidate_max = candidate_max
                best_savings = savings

    return best_candidate