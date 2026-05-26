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
    dist_to_depot = np.linalg.norm(available_customers - depot, axis=1)
    # Identify deciding truck (closest to current_position)
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    deciding_truck_idx = np.argmin(dist_to_current)
    # Compute regrets
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
    # Helper to compute route time from waypoints
    def route_time(waypoints):
        t = 0.0
        for p in range(len(waypoints)-1):
            t += np.linalg.norm(waypoints[p] - waypoints[p+1])
        return t
    # Helper to simulate greedy insertion of a set of customers into routes
    # routes: list of lists of waypoints (each list is a truck's route starting at pos, ending at depot)
    # customers: list of customer positions to insert
    def simulate_greedy(routes, customers):
        for cust in customers:
            best_truck = -1
            best_pos = -1
            best_inc = float('inf')
            for t_idx, waypoints in enumerate(routes):
                # insertion gaps: from first to last segment
                for p in range(len(waypoints)-1):
                    prev = waypoints[p]
                    nxt = waypoints[p+1]
                    inc = np.linalg.norm(prev - cust) + np.linalg.norm(cust - nxt) - np.linalg.norm(prev - nxt)
                    if inc < best_inc:
                        best_inc = inc
                        best_truck = t_idx
                        best_pos = p+1
            routes[best_truck].insert(best_pos, cust)
        # compute max route time
        max_time = 0.0
        for waypoints in routes:
            rt = route_time(waypoints)
            if rt > max_time:
                max_time = rt
        return max_time
    # Determine number of customers to simulate (farthest from depot)
    K = min(5, n_available - 1)  # at most 5 unassigned
    # Evaluate each candidate
    best_cand = None
    best_ttt = float('inf')
    for idx in candidates:
        cand_cust = available_customers[idx]
        # Build routes: deciding truck with candidate first, others just pos->depot
        routes = []
        for t in range(n_trucks):
            if t == deciding_truck_idx:
                route = [current_position, cand_cust, depot]
            else:
                route = [truck_positions[t], depot]
            routes.append(route)
        # Unassigned customers (all except candidate)
        unassigned = [available_customers[i] for i in range(n_available) if i != idx]
        # Pick K farthest from depot
        if len(unassigned) <= K:
            sim_customers = unassigned
        else:
            # sort indices by distance to depot descending
            indices = list(range(len(unassigned)))
            indices.sort(key=lambda i: -np.linalg.norm(unassigned[i] - depot))
            sim_customers = [unassigned[i] for i in indices[:K]]
        ttt = simulate_greedy(routes, sim_customers)
        if ttt < best_ttt:
            best_ttt = ttt
            best_cand = idx
    # Evaluate waiting if more than one truck
    if n_trucks > 1:
        # Routes for other trucks only
        routes_wait = []
        for t in range(n_trucks):
            if t != deciding_truck_idx:
                routes_wait.append([truck_positions[t], depot])
        # Unassigned: all customers
        unassigned_wait = [available_customers[i] for i in range(n_available)]
        # Pick K farthest (or all if less)
        if len(unassigned_wait) <= K:
            sim_wait_customers = unassigned_wait
        else:
            indices = list(range(len(unassigned_wait)))
            indices.sort(key=lambda i: -np.linalg.norm(unassigned_wait[i] - depot))
            sim_wait_customers = [unassigned_wait[i] for i in indices[:K]]
        wait_ttt = simulate_greedy(routes_wait, sim_wait_customers)
        wait_ttt = current_time + wait_ttt  # because routes times are from current_time?
        # Actually simulate_greedy returns total route time from start positions; we need to add current_time?
        # The routes include travel times, so the return time is current_time + max route time.
        # But wait_ttt computed from simulate_greedy already is the max route time. So we add current_time.
        wait_ttt = current_time + wait_ttt
        if best_cand is None or best_ttt >= wait_ttt:
            return None
    # If we have best_cand and (no wait or best_ttt < wait_ttt), return best_cand
    if best_cand is not None:
        return best_cand
    else:
        return None