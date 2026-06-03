import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    # Identify active truck index
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]

    # Distances from active truck to depot
    active_to_depot = np.linalg.norm(current_position - depot_position)

    # Distances from other trucks to depot (for regret computation)
    other_trucks = truck_positions[np.arange(len(truck_positions)) != active_idx]
    if len(other_trucks) > 0:
        max_other_depot_dist = np.max(np.linalg.norm(other_trucks - depot_position, axis=1))
    else:
        max_other_depot_dist = 0.0  # single truck

    # Precompute active to customer and customer to depot
    active_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    active_return = active_to_cust + cust_to_depot

    # Regret: how much active return exceeds max other depot distance
    regret = np.maximum(0, active_return - max_other_depot_dist)

    # Depot pressure: active's distance to depot (scalar), broadcast to customers
    depot_pressure = active_to_depot

    # Combined score
    score = active_return + regret + 0.5 * depot_pressure

    best_idx = np.argmin(score)
    return int(best_idx)