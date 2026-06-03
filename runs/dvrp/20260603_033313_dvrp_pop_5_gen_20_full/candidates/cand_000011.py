import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
) -> int | None:
    n_cust = available_customers.shape[0]
    if n_cust == 0:
        return None

    # Find active truck index (closest to current_position)
    dists_to_trucks = np.linalg.norm(truck_positions - current_position, axis=1)
    active_idx = np.argmin(dists_to_trucks)

    # Helper to estimate makespan for a given state
    def estimate_makespan(truck_pos, cust_set):
        n_rem = cust_set.shape[0]
        if n_rem == 0:
            # All customers served: return max distance from trucks to depot
            return np.max(np.linalg.norm(truck_pos - depot_position, axis=1))
        # Assign each customer to nearest truck (tie: lower index)
        # distances: (n_rem, n_trucks)
        cust_dists = np.linalg.norm(
            cust_set[:, np.newaxis, :] - truck_pos[np.newaxis, :, :], axis=2
        )
        nearest_idx = np.argmin(cust_dists, axis=1)
        # Collect customers per truck
        assignments = [[] for _ in range(truck_pos.shape[0])]
        for ci, ti in enumerate(nearest_idx):
            assignments[ti].append(ci)
        # Compute route length per truck using nearest neighbor heuristic
        total_dists = np.zeros(truck_pos.shape[0])
        for ti in range(truck_pos.shape[0]):
            assigned = assignments[ti]
            if len(assigned) == 0:
                total_dists[ti] = np.linalg.norm(truck_pos[ti] - depot_position)
                continue
            current = truck_pos[ti]
            remaining = set(assigned)
            route_len = 0.0
            while remaining:
                # Find nearest customer among remaining
                rem_list = list(remaining)
                rem_cust = cust_set[rem_list]
                dists = np.linalg.norm(rem_cust - current, axis=1)
                nearest = rem_list[np.argmin(dists)]
                route_len += np.linalg.norm(cust_set[nearest] - current)
                current = cust_set[nearest]
                remaining.remove(nearest)
            route_len += np.linalg.norm(depot_position - current)
            total_dists[ti] = route_len
        return np.max(total_dists)

    # Baseline: no move (waiting)
    base_makespan = estimate_makespan(truck_positions, available_customers)
    best_makespan = base_makespan
    best_idx = None

    # Evaluate each customer as next assignment
    for i in range(n_cust):
        # Move active truck to customer i (i.e., serve it)
        new_truck_pos = truck_positions.copy()
        new_truck_pos[active_idx] = available_customers[i]
        remaining_cust = np.delete(available_customers, i, axis=0)
        makespan = estimate_makespan(new_truck_pos, remaining_cust)
        if makespan < best_makespan:
            best_makespan = makespan
            best_idx = i

    return best_idx