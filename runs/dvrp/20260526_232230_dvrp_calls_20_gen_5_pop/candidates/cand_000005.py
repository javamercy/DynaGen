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

    n_trucks = len(truck_positions)
    # Identify active truck index
    active_idx = np.where(np.all(truck_positions == current_position, axis=1))[0][0]
    other_trucks = np.delete(truck_positions, active_idx, axis=0)

    # Compute base (distance from current to depot)
    base_active = np.linalg.norm(current_position - depot_position)

    # Compute detour for active truck for each customer
    dist_curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    detour_active = dist_curr_to_cust + dist_cust_to_depot - base_active

    if n_trucks == 1:
        # Only one truck, must serve, choose smallest detour
        best_idx = np.argmin(detour_active)
        return int(best_idx)

    # Compute detour for all other trucks (vectorized)
    # For each other truck, distance to each customer + customer to depot - distance to depot
    other_base = np.linalg.norm(other_trucks - depot_position, axis=1)  # (n_other,)
    other_to_cust = np.linalg.norm(other_trucks[:, None] - available_customers[None, :], axis=2)  # (n_other, n_cust)
    detour_other = other_to_cust + dist_cust_to_depot[None, :] - other_base[:, None]  # (n_other, n_cust)
    min_other = np.min(detour_other, axis=0)  # per customer

    # Regret = detour_active - min_other (positive means active worse)
    regret = detour_active - min_other

    # Customers with negative regret (active is best or tied)
    better_mask = regret <= 1e-9  # treat ties as better
    if np.any(better_mask):
        # Among better customers, choose smallest regret (most negative) -> largest advantage
        # If ties, choose smallest detour_active (to minimize travel time)
        better_indices = np.where(better_mask)[0]
        # Sort by regret ascending (most negative first), then by detour ascending
        sorted_indices = better_indices[np.lexsort((detour_active[better_indices], regret[better_indices]))]
        return int(sorted_indices[0])
    else:
        return None