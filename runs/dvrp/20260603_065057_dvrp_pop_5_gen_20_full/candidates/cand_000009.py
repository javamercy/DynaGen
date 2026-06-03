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
        """Simulate greedy minimax assignment for all trucks from current state.
        initial_customer_idx: None for wait, otherwise index into available_customers."""
        n_trucks = len(truck_positions)
        n_customers = len(available_customers)

        # Copy positions and route distances
        positions = truck_positions.copy()
        travel = np.zeros(n_trucks)  # accumulated travel distance for each truck
        remaining = list(range(n_customers))

        # Apply initial action
        if initial_customer_idx is not None:
            remaining.remove(initial_customer_idx)
            travel[truck_idx] = np.linalg.norm(positions[truck_idx] - available_customers[initial_customer_idx])
            positions[truck_idx] = available_customers[initial_customer_idx]

        # Greedy minimax assignment: repeatedly assign to minimize max travel time
        while remaining:
            best_pair = None
            best_max = np.inf
            # Precompute current depot distances for all trucks
            current_depot_dists = np.linalg.norm(positions - depot_position, axis=1)
            current_totals = travel + current_depot_dists  # current max if no more customers
            current_max = np.max(current_totals)
            # For each remaining customer and each truck, compute new max
            for c in remaining:
                for t in range(n_trucks):
                    # New travel and position if truck t takes customer c
                    dist_to_c = np.linalg.norm(positions[t] - available_customers[c])
                    new_travel_t = travel[t] + dist_to_c
                    new_depot_dist_t = np.linalg.norm(available_customers[c] - depot_position)
                    new_total_t = new_travel_t + new_depot_dist_t
                    # The new max for this assignment
                    # For other trucks, use current totals (unchanged)
                    other_max = current_max
                    if t == np.argmax(current_totals):
                        # Need to recompute max without truck t
                        other_vals = np.delete(current_totals, t)
                        other_max = np.max(other_vals) if len(other_vals) > 0 else 0
                    overall_max = max(other_max, new_total_t)
                    if overall_max < best_max:
                        best_max = overall_max
                        best_pair = (t, c)
            # Assign best pair
            t, c = best_pair
            travel[t] += np.linalg.norm(positions[t] - available_customers[c])
            positions[t] = available_customers[c]
            remaining.remove(c)

        # Compute final total distances for each truck
        final_depot_dists = np.linalg.norm(positions - depot_position, axis=1)
        totals = travel + final_depot_dists
        return np.max(totals)

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