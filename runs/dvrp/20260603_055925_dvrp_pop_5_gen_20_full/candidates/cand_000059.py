import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    def dist(a, b):
        return np.linalg.norm(a - b)

    # Current max return time if all trucks return now
    distances_to_depot = np.linalg.norm(truck_positions - depot_position, axis=1)
    current_max = np.max(distances_to_depot)

    best_index = None
    best_change = np.inf
    best_immediate = np.inf

    for i, cust in enumerate(available_customers):
        immediate = dist(current_position, cust) + dist(cust, depot_position)
        change = max(0.0, immediate - current_max)

        if change < best_change - 1e-6:
            best_change = change
            best_immediate = immediate
            best_index = i
        elif abs(change - best_change) < 1e-6 and immediate < best_immediate - 1e-6:
            best_immediate = immediate
            best_index = i

    # Wait if serving any customer increases the max
    if best_change > 0.0:
        return None
    else:
        return best_index