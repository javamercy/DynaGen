import numpy as np

def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None

    # Find index of the deciding truck
    truck_idx = None
    for idx, pos in enumerate(truck_positions):
        if np.array_equal(pos, current_position):
            truck_idx = idx
            break
    if truck_idx is None:
        truck_idx = 0  # fallback (should not happen)

    def simulate(action):
        """Simulate greedy assignment from current state with initial action.
        action: None for wait, else index into available_customers."""
        n_trucks = len(truck_positions)
        cur_pos = truck_positions.copy()
        total_dist = np.zeros(n_trucks)
        remaining = list(range(len(available_customers)))

        # Apply initial action
        if action is not None:
            d = np.linalg.norm(cur_pos[truck_idx] - available_customers[action])
            total_dist[truck_idx] += d
            cur_pos[truck_idx] = available_customers[action]
            remaining.remove(action)

        # Greedy assignment: repeatedly assign nearest customer to any truck
        while remaining:
            best_dist = np.inf
            best_truck = -1
            best_cust = -1
            for t in range(n_trucks):
                for c in remaining:
                    d = np.linalg.norm(cur_pos[t] - available_customers[c])
                    if d < best_dist:
                        best_dist = d
                        best_truck = t
                        best_cust = c
            # Assign
            total_dist[best_truck] += best_dist
            cur_pos[best_truck] = available_customers[best_cust]
            remaining.remove(best_cust)

        # Add return to depot
        for t in range(n_trucks):
            total_dist[t] += np.linalg.norm(cur_pos[t] - depot_position)

        return np.max(total_dist)

    best_score = np.inf
    best_action = None

    # Evaluate wait
    wait_score = simulate(None)
    if wait_score < best_score:
        best_score = wait_score
        best_action = None

    # Evaluate each customer
    for i in range(len(available_customers)):
        score = simulate(i)
        if score < best_score:
            best_score = score
            best_action = i

    return best_action