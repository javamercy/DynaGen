def choose_next_customer(current_position, depot_position, truck_positions, available_customers):
    if len(available_customers) == 0:
        return None
    # Lower bound return time for other trucks (distance to depot)
    other_returns = [
        np.linalg.norm(pos - depot_position)
        for pos in truck_positions
        if not np.allclose(pos, current_position)
    ]
    max_other_return = max(other_returns, default=0.0)
    
    def route_time(waypoints):
        total = 0.0
        prev = current_position
        for wp in waypoints:
            total += np.linalg.norm(wp - prev)
            prev = wp
        total += np.linalg.norm(depot_position - prev)
        return total
    
    n_avail = len(available_customers)
    best_idx = None
    best_ttt = float('inf')
    best_first_dist = float('inf')
    
    if n_avail <= 20:  # depth-2 enumeration
        for i in range(n_avail):
            first = available_customers[i]
            first_dist = np.linalg.norm(first - current_position)
            # Evaluate with no second customer
            active_return = route_time([first])
            ttt = max(active_return, max_other_return)
            if ttt < best_ttt or (ttt == best_ttt and first_dist < best_first_dist):
                best_ttt = ttt
                best_idx = i
                best_first_dist = first_dist
            # Evaluate with each possible second customer
            for j in range(n_avail):
                if j == i:
                    continue
                second = available_customers[j]
                route = [first, second]
                active_return = route_time(route)
                ttt = max(active_return, max_other_return)
                if ttt < best_ttt or (ttt == best_ttt and first_dist < best_first_dist):
                    best_ttt = ttt
                    best_idx = i
                    best_first_dist = first_dist
    else:  # depth-1
        for i in range(n_avail):
            first = available_customers[i]
            first_dist = np.linalg.norm(first - current_position)
            active_return = route_time([first])
            ttt = max(active_return, max_other_return)
            if ttt < best_ttt or (ttt == best_ttt and first_dist < best_first_dist):
                best_ttt = ttt
                best_idx = i
                best_first_dist = first_dist
    return best_idx