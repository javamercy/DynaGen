import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    n_available = available_customers.shape[0]
    if n_available == 0:
        return None
    n_trucks = truck_positions.shape[0]
    if n_trucks == 1:
        d_curr = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(d_curr))
    # identify current truck index (by closest position)
    diffs = truck_positions - current_position
    curr_idx = np.argmin(np.linalg.norm(diffs, axis=1))
    # distances from current truck to depot
    d_curr_to_depot = np.linalg.norm(current_position - depot_position)
    # distances from other trucks to depot
    d_trucks_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    other_to_depot = np.delete(d_trucks_to_depot, curr_idx)
    other_max_remaining = np.max(other_to_depot) if other_to_depot.size > 0 else 0.0
    # precompute distances from each other truck to each customer
    # shape: (n_trucks-1, n_available)
    other_mask = np.ones(n_trucks, dtype=bool)
    other_mask[curr_idx] = False
    other_pos = truck_positions[other_mask]  # (n_other, 2)
    # distances from each other truck to each customer
    other_to_cust = np.linalg.norm(other_pos[:, np.newaxis, :] - available_customers[np.newaxis, :, :], axis=2)  # (n_other, n_avail)
    min_other_dist = np.min(other_to_cust, axis=0)  # (n_avail,)
    # distances from current truck to customers
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)  # (n_avail,)
    # distances from customers to depot
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)  # (n_avail,)
    curr_comp = curr_to_cust + cust_to_depot
    regret = curr_to_cust - min_other_dist
    new_max = np.maximum(curr_comp, other_max_remaining)
    cost = new_max + 0.1 * regret
    best_cust_idx = np.argmin(cost)
    best_cust_cost = cost[best_cust_idx]
    waiting_cost = max(d_curr_to_depot, other_max_remaining)
    if best_cust_cost < waiting_cost:
        return int(best_cust_idx)
    else:
        return None