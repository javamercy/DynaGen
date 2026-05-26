import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    depot = depot_position
    current_pos = current_position
    other_trucks = truck_positions
    # Precompute other trucks' current return times (distance to depot)
    other_returns = dist(other_trucks, depot)
    
    w_max = 1.0
    w_mean = 0.2
    w_depot = 0.2

    best_idx = None
    best_score = float('inf')
    best_active_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_pos, customer)
        d_cust_depot = dist(customer, depot)
        active_return = d_truck_cust + d_cust_depot
        # Total return times for all trucks
        all_returns = np.concatenate(([active_return], other_returns))
        new_max = np.max(all_returns)
        new_mean = np.mean(all_returns)
        score = w_max * new_max + w_mean * new_mean + w_depot * d_cust_depot
        # Tie-breaking: lower active_return preferred
        if score < best_score or (score == best_score and active_return < best_active_return):
            best_score = score
            best_active_return = active_return
            best_idx = i

    return best_idx