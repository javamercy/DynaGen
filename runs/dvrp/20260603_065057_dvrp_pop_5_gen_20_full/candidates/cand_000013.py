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
        # skip the current truck's position? We include all; since current truck is at current_position, its contribution would be identical to tour_time_now, but we want best other, so we can exclude it. However, truck_positions includes all trucks. We'll just compute for all, but the current truck's tour_time_now will be computed again. To avoid self-comparison, we can exclude the current position. But we don't know which index corresponds to current_position. Since we compute for all, the best_other_tour might consider the current truck's own tour if it is better, which is unfair. So we should exclude current_position from the set of other trucks. We can do this by building a list of other positions.
    other_positions = [pos for pos in truck_positions if not np.array_equal(pos, current_position)]
    for pos in other_positions:
        dist_other_to_cust = np.linalg.norm(available_customers - pos, axis=1)
        other_tour = dist_other_to_cust + dist_cust_to_depot
        best_other_tour = np.minimum(best_other_tour, other_tour)

    # regret = tour_time_now - best_other_tour
    regret = tour_time_now - best_other_tour

    # If all regrets are positive, defer (return None)
    if np.all(regret > 0):
        return None

    # Among customers with regret <= 0, pick the one with smallest tour_time_now
    candidates = np.where(regret <= 0)[0]
    best_idx = candidates[np.argmin(tour_time_now[candidates])]
    return int(best_idx)