import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    n_trucks = len(truck_positions)
    if n_trucks == 1:
        distances = np.linalg.norm(available_customers - current_position, axis=1)
        return int(np.argmin(distances))

    d_trucks_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_idx = int(np.argmin(np.linalg.norm(truck_positions - current_position, axis=1)))
    d_curr_to_depot = d_trucks_to_depot[current_idx]
    other_depot_distances = np.delete(d_trucks_to_depot, current_idx)
    other_max = np.max(other_depot_distances) if len(other_depot_distances) > 0 else -np.inf
    current_max = max(d_curr_to_depot, other_max)

    d_curr_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    d_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    totals = d_curr_to_cust + d_cust_to_depot

    feasible = totals <= current_max
    if not np.any(feasible):
        return None
    feasible_indices = np.where(feasible)[0]
    feasible_totals = totals[feasible_indices]
    best_local_idx = np.argmin(feasible_totals)
    return int(feasible_indices[best_local_idx])