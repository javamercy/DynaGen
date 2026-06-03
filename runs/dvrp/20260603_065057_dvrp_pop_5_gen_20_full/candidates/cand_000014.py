import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    # distances from current truck to each customer
    dist_current_to_cust = np.linalg.norm(available_customers - current_position, axis=1)
    # distances from each customer to depot
    dist_cust_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    # tour time if served now then return to depot
    tour_time_now = dist_current_to_cust + dist_cust_to_depot

    # for each other truck, compute distance to each customer + customer to depot
    best_other_tour = np.full(len(available_customers), np.inf)
    for pos in truck_positions:
        dist_other_to_cust = np.linalg.norm(available_customers - pos, axis=1)
        other_tour = dist_other_to_cust + dist_cust_to_depot
        best_other_tour = np.minimum(best_other_tour, other_tour)

    # regret = tour_time_now - best_other_tour
    regret = tour_time_now - best_other_tour

    # Customers where current truck has an advantage (regret <= 0)
    candidates = np.where(regret <= 0)[0]
    if len(candidates) == 0:
        return None

    # Among those, pick the one closest to current position
    best_idx = candidates[np.argmin(dist_current_to_cust[candidates])]
    return int(best_idx)