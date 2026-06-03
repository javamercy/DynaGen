import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    # distances from active truck to customers
    dist_active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from customers to depot
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # completion time for active truck if it serves customer and returns
    T_active = dist_active_to_cust + dist_cust_to_depot

    n_trucks = truck_positions.shape[0]
    if n_trucks == 1:
        best_idx = np.argmin(T_active)
        return int(best_idx)

    # find active truck index
    is_active = np.all(np.isclose(truck_positions, current_position), axis=1)
    active_idx = np.where(is_active)[0][0]

    # distances from other trucks to depot
    dist_other_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    dist_other_to_depot[active_idx] = -np.inf  # exclude active
    other_max = np.max(dist_other_to_depot)

    # new makespan if active serves each customer
    new_makespan = np.maximum(T_active, other_max)
    min_makespan = np.min(new_makespan)
    # indices achieving min
    candidate_indices = np.where(new_makespan == min_makespan)[0]
    if len(candidate_indices) == 1:
        best_idx = candidate_indices[0]
    else:
        # tie-break by smallest T_active
        best_idx = candidate_indices[np.argmin(T_active[candidate_indices])]

    return int(best_idx)