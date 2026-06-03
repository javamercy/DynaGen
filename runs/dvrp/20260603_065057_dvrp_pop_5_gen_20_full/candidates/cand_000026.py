import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if available_customers.shape[0] == 0:
        return None
    n_trucks = truck_positions.shape[0]
    d_curr = np.linalg.norm(available_customers - current_position, axis=1)
    if n_trucks == 1:
        return int(np.argmin(d_curr))
    # compute min distance from any other truck to each customer
    all_dists = np.linalg.norm(truck_positions[:, np.newaxis, :] - available_customers, axis=2)
    # find index of current truck
    current_idx = np.argmin(np.linalg.norm(truck_positions - current_position, axis=1))
    all_dists[current_idx, :] = np.inf
    min_other_dist = np.min(all_dists, axis=0)
    regret = d_curr - min_other_dist
    positive_mask = regret > 0
    if np.any(positive_mask):
        # among positive regrets, pick the smallest (most advantageous)
        best_idx = np.argmin(np.where(positive_mask, regret, np.inf))
    else:
        # all regrets non-positive, pick the one with smallest regret (least negative)
        best_idx = np.argmin(regret)
    return int(best_idx)