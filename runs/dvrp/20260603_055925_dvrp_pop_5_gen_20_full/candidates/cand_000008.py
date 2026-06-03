import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    n_trucks = len(truck_positions)
    # Identify active truck index
    active_idx = None
    for i, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            active_idx = i
            break
    if active_idx is None:
        # Fallback: treat as any truck (should not happen per interface)
        active_idx = 0

    # Current makespan: max return time among all trucks (using current positions)
    current_max = max(dist(pos, depot_position) for pos in truck_positions)

    best_score = -float('inf')
    best_new_return = float('inf')
    best_idx = None

    for i, cust in enumerate(available_customers):
        new_return = dist(current_position, cust) + dist(cust, depot_position)

        if n_trucks == 1:
            # Single truck: must serve; score is negative of new_return (higher = shorter)
            score = -new_return
        else:
            # Compute max return among other trucks (excluding active)
            max_other = max(
                dist(truck_positions[j], depot_position)
                for j in range(n_trucks) if j != active_idx
            )
            new_max = max(max_other, new_return)
            score = current_max - new_max  # positive = improvement

        if score > best_score or (score == best_score and new_return < best_new_return):
            best_score = score
            best_new_return = new_return
            best_idx = i

    if n_trucks == 1:
        # Always serve the customer with best (i.e., smallest new_return)
        return best_idx
    else:
        # Only serve if makespan does not increase (score >= 0)
        if best_score >= 0 and best_idx is not None:
            return best_idx
        else:
            return None