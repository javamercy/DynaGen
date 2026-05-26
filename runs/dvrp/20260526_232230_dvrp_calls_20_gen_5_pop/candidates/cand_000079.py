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
    other_trucks = truck_positions
    # Current return time for this truck if it returns now
    current_truck_return = dist(current_pos, depot)
    # Returns of other trucks (distance to depot)
    other_returns = dist(other_trucks, depot)
    n_trucks = len(truck_positions)
    current_max_return = max(current_truck_return, np.max(other_returns))

    best_idx = None
    best_score = float('inf')
    best_new_return = float('inf')

    for i, customer in enumerate(available_customers):
        d_truck_cust = dist(current_pos, customer)
        d_cust_depot = dist(customer, depot)
        new_return = d_truck_cust + d_cust_depot
        # All returns: new_return plus other returns
        all_returns = np.concatenate([[new_return], other_returns])
        new_max = np.max(all_returns)
        new_mean = np.mean(all_returns)
        new_std = np.std(all_returns)
        # Weighted score: maximize balance
        w_max = 1.0
        w_mean = 2.0
        w_var = 1.0
        score = w_max * new_max + w_mean * new_mean + w_var * new_std
        if score < best_score or (score == best_score and new_return < best_new_return):
            best_score = score
            best_new_return = new_return
            best_idx = i

    # Evaluate whether waiting could be better
    # Compute new_max if best customer is chosen
    best_customer = available_customers[best_idx]
    d_truck_cust = dist(current_pos, best_customer)
    d_cust_depot = dist(best_customer, depot)
    best_new_return = d_truck_cust + d_cust_depot
    best_new_max = max(best_new_return, np.max(other_returns))
    # Wait if serving best customer would increase max by more than 15%
    if best_new_max > current_max_return * 1.15:
        return None
    return best_idx