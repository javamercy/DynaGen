import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    if len(available_customers) == 0:
        return None
    centroid = np.mean(available_customers, axis=0)
    dist_current = np.linalg.norm(available_customers - current_position, axis=1)
    dist_centroid = np.linalg.norm(available_customers - centroid, axis=1)
    alpha = 0.5
    scores = dist_current + alpha * dist_centroid
    best_idx = int(np.argmin(scores))
    return best_idx