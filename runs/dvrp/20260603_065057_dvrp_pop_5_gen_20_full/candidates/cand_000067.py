import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    
    # Identify the index of the current truck in truck_positions
    truck_idx = None
    for i, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            truck_idx = i
            break
    if truck_idx is None:
        # fallback: use nearest truck (should not happen normally)
        dists = np.linalg.norm(truck_positions - current_position, axis=1)
        truck_idx = int(np.argmin(dists))
    
    n_cust = len(available_customers)
    # Compute distance from each truck to each customer
    dist_truck_cust = np.linalg.norm(available_customers[np.newaxis, :, :] - truck_positions[:, np.newaxis, :], axis=2)  # (n_trucks, n_cust)
    # Distance from each customer to depot
    dist_cust_depot = np.linalg.norm(available_customers - depot_position, axis=1)  # (n_cust,)
    
    # Total cost for each truck-customer pair (truck to customer + customer to depot)
    total_costs = dist_truck_cust + dist_cust_depot[np.newaxis, :]  # (n_trucks, n_cust)
    
    # Cost for current truck
    this_cost = total_costs[truck_idx, :]  # (n_cust,)
    # Best cost among other trucks
    other_costs = np.delete(total_costs, truck_idx, axis=0)  # (n_trucks-1, n_cust)
    best_other_cost = np.min(other_costs, axis=0)  # (n_cust,)
    
    # Regret: advantage of current truck over the best other
    regret = best_other_cost - this_cost  # (n_cust,)
    
    # Only consider positive regret
    if np.max(regret) <= 0:
        return None
    
    # Select customer with maximum regret; tie-break with smallest distance to depot
    max_regret = np.max(regret)
    candidates = np.where(regret >= max_regret - 1e-9)[0]
    if len(candidates) == 1:
        return int(candidates[0])
    else:
        # Tie-break by smallest distance to depot
        tie_idx = candidates[np.argmin(dist_cust_depot[candidates])]
        return int(tie_idx)