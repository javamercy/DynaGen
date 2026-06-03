import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    all_direct = np.array([np.linalg.norm(truck - depot_position) for truck in truck_positions])
    current_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    others_mask = np.ones(len(truck_positions), dtype=bool)
    others_mask[current_idx] = False
    other_returns_max = np.max(all_direct[others_mask]) if any(others_mask) else 0.0
    current_direct = all_direct[current_idx]
    M0 = max(current_direct, other_returns_max)
    best_idx = None
    best_M = float('inf')
    best_cost_now = float('inf')
    for i, cust in enumerate(available_customers):
        cost_now = np.linalg.norm(current_position - cust) + np.linalg.norm(cust - depot_position)
        M_i = max(cost_now, other_returns_max)
        if M_i < best_M or (M_i == best_M and cost_now < best_cost_now):
            best_M = M_i
            best_cost_now = cost_now
            best_idx = i
    if best_M > M0:
        return None
    return best_idx