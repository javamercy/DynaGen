def choose_next_customer(
    current_position: np.ndarray,
    depot_position: np.ndarray,
    truck_positions: np.ndarray,
    available_customers: np.ndarray,
    current_time: float,
) -> int | None:
    if len(available_customers) == 0:
        return None

    n_trucks = len(truck_positions)
    best_index = None
    best_ttt = float('inf')

    # Identify the index of the deciding truck
    deciding_truck_idx = None
    for i, pos in enumerate(truck_positions):
        if np.allclose(pos, current_position, atol=1e-6):
            deciding_truck_idx = i
            break
    if deciding_truck_idx is None:
        deciding_truck_idx = 0  # fallback

    # Candidates: indices of available customers plus None for waiting
    candidates = list(range(len(available_customers))) + [None]

    for cand in candidates:
        # Initialize routes: each route is a list of positions starting at truck position and ending at depot
        routes = [ [truck_positions[t].copy(), depot_position] for t in range(n_trucks) ]

        # Build list of unassigned customer positions (excluding the candidate if specified)
        unassigned = []
        for i, cust in enumerate(available_customers):
            if cand is None or i != cand:
                unassigned.append(cust.copy())

        # If candidate is a customer, insert it as the first stop for the deciding truck
        if cand is not None:
            cand_pos = available_customers[cand].copy()
            route = routes[deciding_truck_idx]
            # Insert after the start (index 0) and before depot (index 1)
            routes[deciding_truck_idx] = [route[0], cand_pos, route[1]]

        # Greedily assign remaining customers: sort by distance to depot descending
        unassigned.sort(key=lambda c: -np.linalg.norm(c - depot_position))

        for cust in unassigned:
            best_truck = -1
            best_pos = -1
            best_inc = float('inf')
            for t_idx in range(n_trucks):
                waypoints = routes[t_idx]
                # Determine allowed insertion positions (gaps between consecutive waypoints)
                # For the deciding truck with a fixed first stop, only insert after that stop
                start_gap = 0
                if cand is not None and t_idx == deciding_truck_idx:
                    start_gap = 1  # skip the gap between start and first stop
                for p in range(start_gap, len(waypoints)-1):
                    prev = waypoints[p]
                    nxt = waypoints[p+1]
                    inc = np.linalg.norm(prev - cust) + np.linalg.norm(cust - nxt) - np.linalg.norm(prev - nxt)
                    if inc < best_inc:
                        best_inc = inc
                        best_truck = t_idx
                        best_pos = p+1  # insert after prev (i.e., at position p+1 in list)
            # Insert customer at the best position
            routes[best_truck].insert(best_pos, cust)

        # Compute maximum route time across all trucks
        max_route_time = 0.0
        for waypoints in routes:
            route_time = 0.0
            for i in range(len(waypoints)-1):
                route_time += np.linalg.norm(waypoints[i] - waypoints[i+1])
            if route_time > max_route_time:
                max_route_time = route_time
        ttt = current_time + max_route_time

        if ttt < best_ttt:
            best_ttt = ttt
            best_index = cand

    return best_index