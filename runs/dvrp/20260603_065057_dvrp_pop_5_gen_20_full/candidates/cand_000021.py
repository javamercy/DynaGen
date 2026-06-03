import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None

    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_to_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    tour_time_now = dist_current + dist_to_depot

    best_other_tour = np.full(len(available_customers), np.inf)
    for pos in truck_positions:
        dist_other = np.linalg.norm(available_customers - pos, axis=1)
        other_tour = dist_other + dist_to_depot
        best_other_tour = np.minimum(best_other_tour, other_tour)

    regret = tour_time_now - best_other_tour

    if np.any(regret <= 0):
        candidates = np.where(regret <= 0)[0]
        best_idx = candidates[np.argmin(regret[candidates])]
        return int(best_idx)
    else:
        best_idx = np.argmin(regret)
        return int(best_idx)