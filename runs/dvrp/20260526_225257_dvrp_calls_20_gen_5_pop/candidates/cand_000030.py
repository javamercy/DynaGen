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
    d_me = np.linalg.norm(available_customers - current_position, axis=1)
    d_depot = np.linalg.norm(available_customers - depot_position, axis=1)
    own_score = d_me + d_depot

    other_trucks = [p for i, p in enumerate(truck_positions) if not np.allclose(p, current_position)]
    if not other_trucks:
        return int(np.argmin(own_score))

    d_other = np.linalg.norm(available_customers[:, None, :] - np.array(other_trucks)[None, :, :], axis=2).min(axis=1)
    regret = d_other - d_me
    gamma = 0.8
    score = own_score - gamma * regret
    best_idx = np.argmin(score)
    return int(best_idx)