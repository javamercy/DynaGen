import numpy as np
from typing import Optional

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> Optional[int]:
    if available_customers.shape[0] == 0:
        return None
    # distances from current to each customer
    curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
    cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    total = curr_to_cust + cust_to_depot
    # depot distances for all trucks
    depot_dists = np.linalg.norm(truck_positions - depot_position, axis=1)
    # find active truck index
    active_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    # max of other trucks' depot distances
    other_dists = np.delete(depot_dists, active_idx)
    max_other = np.max(other_dists) if other_dists.size > 0 else 0.0
    # resulting max if we go to each customer and then return
    resulting_max = np.maximum(max_other, total)
    best_idx = np.argmin(resulting_max)
    return int(best_idx)