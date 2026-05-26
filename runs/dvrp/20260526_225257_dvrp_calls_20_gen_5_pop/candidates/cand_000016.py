import numpy as np

def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    n_available = len(available_customers)
    if n_available == 0:
        return None

    n_trucks = truck_positions.shape[0]
    depot = depot_position

    # Identify deciding truck index (closest to current_position)
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    deciding_truck_idx = np.argmin(dist_to_current)

    # Precompute distances
    dist_to_depot = np.linalg.norm(available_customers - depot, axis=1)

    # Compute regrets (non-positive means current truck is best)
    regrets = np.empty(n_available)
    for i in range(n_available):
        cust = available_customers[i]
        d_curr_cust = np.linalg.norm(current_position - cust)
        cost_now = current_time + d_curr_cust + dist_to_depot[i]
        best_other = float('inf')
        for j in range(n_trucks):
            if j == deciding_truck_idx:
                continue
            d_truck_cust = np.linalg.norm(truck_positions[j] - cust)
            cost_other = current_time + d_truck_cust + dist_to_depot[i]
            if cost_other < best_other:
                best_other = cost_other
        if n_trucks == 1:
            regrets[i] = -float('inf')
        else:
            regrets[i] = cost_now - best_other

    candidates = [i for i in range(n_available) if regrets[i] <= 0]
    if not candidates:
        return None

    # Helper to compute route time
    def route_time(waypoints):
        t = 0.0
        for p in range(len(waypoints)-1):
            t += np.linalg.norm(waypoints[p] - waypoints[p+1])
        return t

    # Select high-impact customers for lookahead (farthest from depot)
    unassigned_indices = list(range(n_available))
    # Sort customers by distance to depot descending
    sorted_by_dist = np.argsort(-dist_to_depot)
    K = min(5, n_available)  # number of high-impact customers to simulate

    best_index = None
    best_ttt = float('inf')

    for cand_idx in candidates:
        cand_cust = available_customers[cand_idx]
        # Initialize routes: each route starts at truck position and ends at depot
        routes = [ [truck_positions[t].copy(), depot] for t in range(n_trucks) ]
        # Insert candidate as first stop for deciding truck
        routes[deciding_truck_idx] = [routes[deciding_truck_idx][0], cand_cust, routes[deciding_truck_idx][1]]

        # Determine high-impact customers to assign (excluding candidate)
        high_impact = []
        for idx in sorted_by_dist:
            if idx == cand_idx:
                continue
            high_impact.append(available_customers[idx].copy())
            if len(high_impact) >= K:
                break

        # Greedy assignment of high-impact customers
        for cust in high_impact:
            best_truck = -1
            best_pos = -1
            best_inc = float('inf')
            for t_idx in range(n_trucks):
                waypoints = routes[t_idx]
                if t_idx == deciding_truck_idx:
                    start_gap = 1  # skip first segment start->cand_cust
                else:
                    start_gap = 0
                for p in range(start_gap, len(waypoints)-1):
                    prev = waypoints[p]
                    nxt = waypoints[p+1]
                    inc = np.linalg.norm(prev - cust) + np.linalg.norm(cust - nxt) - np.linalg.norm(prev - nxt)
                    if inc < best_inc:
                        best_inc = inc
                        best_truck = t_idx
                        best_pos = p+1
            routes[best_truck].insert(best_pos, cust)

        # Compute max route time
        max_route_time = 0.0
        for waypoints in routes:
            rt = route_time(waypoints)
            if rt > max_route_time:
                max_route_time = rt
        ttt = current_time + max_route_time

        if ttt < best_ttt:
            best_ttt = ttt
            best_index = cand_idx

    # Compute waiting TTT heuristic
    max_truck_to_depot = np.max(np.linalg.norm(truck_positions - depot, axis=1))
    avg_cust_to_depot = np.mean(dist_to_depot)
    waiting_ttt = current_time + max_truck_to_depot + (avg_cust_to_depot * n_available / n_trucks) * 0.3

    if best_ttt > waiting_ttt:
        return None
    else:
        return best_index