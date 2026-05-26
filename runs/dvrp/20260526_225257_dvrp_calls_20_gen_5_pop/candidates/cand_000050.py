import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None
    
    # cost for this truck: travel to customer + return to depot
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + customer_to_depot
    
    # identify other trucks (exclude current position)
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        best_idx = int(np.argmin(cost_now))
        return best_idx
    
    # best other cost for each customer
    other_to_customer = np.linalg.norm(
        available_customers[:, None, :] - other_trucks[None, :, :], axis=2
    )
    best_other_cost = np.min(other_to_customer, axis=1) + customer_to_depot
    
    regret = best_other_cost - cost_now
    max_regret = np.max(regret)
    
    if max_regret > 1e-6:
        # Diversified selection: among customers with regret within 90% of max, pick the one with largest cost_now
        threshold = 0.9 * max_regret
        candidates = np.where(regret >= threshold)[0]
        # Among candidates, pick the one with maximum cost_now (exploration)
        best_idx = int(candidates[np.argmax(cost_now[candidates])])
        return best_idx
    
    # No positive regret: consider waiting
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    is_closest_to_depot = current_depot_dist < np.min(other_depot_dists) - 1e-6
    
    # Increased wait threshold: wait if closest to depot and <=3 remaining customers
    if is_closest_to_depot and len(available_customers) <= 3:
        return None
    else:
        best_idx = int(np.argmin(cost_now))
        return best_idx