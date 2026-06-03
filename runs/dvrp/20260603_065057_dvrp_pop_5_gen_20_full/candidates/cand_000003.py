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
        truck_idx = 0  # fallback

    def simulate_ttt(initial_customer_idx):
        """Simulate greedy assignment for all trucks from current state.
        initial_customer_idx: None for wait, otherwise index into available_customers."""
        n_trucks = len(truck_positions)
        n_customers = len(available_customers)

        # Copy positions and routes
        positions = truck_positions.copy()
        routes = [[] for _ in range(n_trucks)]
        remaining = list(range(n_customers))

        # Apply initial action
        if initial_customer_idx is not None:
            routes[truck_idx].append(initial_customer_idx)
            positions[truck_idx] = available_customers[initial_customer_idx]
            remaining.remove(initial_customer_idx)

        # Greedy assignment: repeatedly assign nearest customer to any truck
        while remaining:
            best_dist = np.inf
            best_truck = None
            best_cust = None
            for t in range(n_trucks):
                for c in remaining:
                    dist = np.linalg.norm(positions[t] - available_customers[c])
                    if dist < best_dist:
                        best_dist = dist
                        best_truck = t
                        best_cust = c
            # Assign
            routes[best_truck].append(best_cust)
            positions[best_truck] = available_customers[best_cust]
            remaining.remove(best_cust)

        # Compute total travel distance for each truck
        total_distances = np.zeros(n_trucks)
        for t in range(n_trucks):
            if routes[t]:
                current = truck_positions[t]
                total = 0.0
                for c in routes[t]:
                    next_pos = available_customers[c]
                    total += np.linalg.norm(current - next_pos)
                    current = next_pos
                total += np.linalg.norm(current - depot_position)
                total_distances[t] = total
            else:
                total_distances[t] = np.linalg.norm(truck_positions[t] - depot_position)

        return np.max(total_distances)

    # Evaluate each customer
    best_score = np.inf
    best_action = None

    for i in range(len(available_customers)):
        score = simulate_ttt(i)
        if score < best_score:
            best_score = score
            best_action = i

    # Evaluate wait
    wait_score = simulate_ttt(None)
    if wait_score < best_score:
        return None
    else:
        return best_action