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

    current_pos = current_position
    depot = depot_position
    other_returns = dist(truck_positions, depot)
    current_max_return = np.max(other_returns)

    best_idx = None
    best_score = float('inf')
    best_new_return = float('inf')

    lambda_penalty = 0.3

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_pos, customer)
        d_cust_depot = dist(customer, depot)
        new_return = d_truck_cust + d_cust_depot
        if new_return > current_max_return:
            penalty = lambda_penalty * (new_return - current_max_return)
            score = new_return + penalty
        else:
            score = new_return
        if score < best_score or (score == best_score and new_return < best_new_return):
            best_score = score
            best_new_return = new_return
            best_idx = i

    return best_idx