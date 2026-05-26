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
    
    # Current truck's cost to serve each customer: travel from current to customer, then to depot
    current_to_customer = np.linalg.norm(available_customers - current_position, axis=1)
    customer_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    cost_now = current_to_customer + customer_to_depot
    
    # Identify other trucks (exclude current position)
    mask = np.all(np.abs(truck_positions - current_position) < 1e-8, axis=1)
    other_trucks = truck_positions[~mask]
    
    if len(other_trucks) == 0:
        # Only one truck, pick customer with minimal cost
        best_idx = np.argmin(cost_now)
        return int(best_idx)
    
    # For each customer, compute best cost if served by another truck (now)
    # For each other truck, cost = distance(other_truck, customer) + distance(customer, depot)
    other_to_customer = np.linalg.norm(available_customers[:, None, :] - other_trucks[None, :, :], axis=2)  # (n_cust, n_other)
    best_other_cost = np.min(other_to_customer, axis=1) + customer_to_depot
    
    # Regret = how much better is it to serve now by this truck vs best other?
    regret = best_other_cost - cost_now  # positive means urgent to serve now
    
    max_regret = np.max(regret)
    if max_regret > 1e-6:
        # There is at least one customer with positive regret
        best_idx = np.argmax(regret)
        return int(best_idx)
    
    # No positive regret: all customers are as good or better served by another truck.
    # Decide whether to wait. Wait if this truck is not the farthest from depot,
    # because if it is farthest, delaying may worsen TTT.
    current_depot_dist = np.linalg.norm(current_position - depot_position)
    other_depot_dists = np.linalg.norm(other_trucks - depot_position, axis=1)
    if current_depot_dist < np.max(other_depot_dists):
        # Current truck is not the farthest, so it can wait (other trucks are further out)
        return None
    else:
        # Current truck is farthest; it should take a customer with minimal cost to start returning
        best_idx = np.argmin(cost_now)
        return int(best_idx)