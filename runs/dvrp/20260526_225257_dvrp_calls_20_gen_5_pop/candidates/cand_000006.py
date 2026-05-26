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

    # Identify the deciding truck index (closest to current_position)
    dist_to_current = np.linalg.norm(truck_positions - current_position, axis=1)
    deciding_truck_idx = np.argmin(dist_to_current)

    # Precompute distances from each truck to each customer and to depot
    depot = depot_position
    dist_to_depot = np.linalg.norm(available_customers - depot, axis=1)

    # For each customer, compute regret = cost_now - best_other_cost
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
            regrets[i] = -float('inf')  # current truck is the only option
        else:
            regrets[i] = cost_now - best_other

    # Candidates: non-positive regret
    candidates = [i for i in range(n_available) if regrets[i] <= 0]
    if not candidates:
        return None

    # Helper to compute route time given list of waypoints
    def route_time(waypoints):
        t = 0.0
        for p in range(len(waypoints)-1):
            t += np.linalg.norm(waypoints[p] - waypoints[p+1])
        return t

    best_index = None
    best_ttt = float('inf')

    # Evaluate each candidate with a greedy insertion of remaining customers
    for cand_idx in candidates:
        cand_cust = available_customers[cand_idx]
        # Initialize routes: each route starts at truck position and ends at depot
        routes = [ [truck_positions[t].copy(), depot] for t in range(n_trucks) ]
        # Insert candidate as first stop for deciding truck
        routes[deciding_truck_idx] = [routes[deciding_truck_idx][0], cand_cust, routes[deciding_truck_idx][1]]

        # Unassigned customers (all except candidate)
        unassigned = [available_customers[i].copy() for i in range(n_available) if i != cand_idx]

        # Greedy assignment: for each unassigned customer, find cheapest insertion across all trucks
        for cust in unassigned:
            best_truck = -1
            best_pos = -1
            best_inc = float('inf')
            for t_idx in range(n_trucks):
                waypoints = routes[t_idx]
                # Determine allowed insertion gaps for deciding truck: can't insert before the first stop (cand_cust)
                start_gap = 0
                # Actually, we can insert anywhere except between start and its first stop which is already fixed?
                # But the decision is to go to that customer first, so we should not insert before it.
                # However, after assigning candidate, we can insert further customers between cand_cust and depot or after?
                # Since the route is start -> cand_cust -> ... -> depot, we can insert after cand_cust.
                # So for deciding truck, start_gap = 1 (skip the first segment start->cand_cust)
                if t_idx == deciding_truck_idx:
                    start_gap = 1
                for p in range(start_gap, len(waypoints)-1):
                    prev = waypoints[p]
                    nxt = waypoints[p+1]
                    inc = np.linalg.norm(prev - cust) + np.linalg.norm(cust - nxt) - np.linalg.norm(prev - nxt)
                    if inc < best_inc:
                        best_inc = inc
                        best_truck = t_idx
                        best_pos = p+1  # insert after prev
            # Insert the customer at best position
            routes[best_truck].insert(best_pos, cust)

        # Compute maximum route time
        max_route_time = 0.0
        for waypoints in routes:
            rt = route_time(waypoints)
            if rt > max_route_time:
                max_route_time = rt
        ttt = current_time + max_route_time

        if ttt < best_ttt:
            best_ttt = ttt
            best_index = cand_idx

    return best_index