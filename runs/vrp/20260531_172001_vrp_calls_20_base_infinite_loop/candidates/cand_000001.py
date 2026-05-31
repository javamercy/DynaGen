import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix.astype(float)
    
    # Initial routes: each customer as its own route
    routes = []
    for i in range(1, n):
        route = [0, i, 0]
        first = i
        last = i
        dist0first = dist[0][i]
        distlast0 = dist[i][0]
        total = dist[0][i] + dist[i][0]
        routes.append({
            'route': route,
            'first': first,
            'last': last,
            'dist0first': dist0first,
            'distlast0': distlast0,
            'total': total
        })
    
    # Add empty routes if needed
    while len(routes) < truck_count:
        routes.append({'route': [0, 0], 'first': -1, 'last': -1, 'dist0first': 0.0, 'distlast0': 0.0, 'total': 0.0})
    
    # Helper to compute max distance among current routes (only non-empty)
    def current_max():
        return max(r['total'] for r in routes) if routes else 0.0
    
    # Greedy merging until we have exactly truck_count routes (if we have more)
    while len(routes) > truck_count:
        # Precompute max for other routes
        totals = [r['total'] for r in routes]
        best_max = float('inf')
        best_i = best_j = -1
        best_merge = None  # (new_route, new_first, new_last, new_dist0first, new_distlast0, new_total)
        for i in range(len(routes)):
            if routes[i]['first'] == -1:  # empty route, skip
                continue
            for j in range(i+1, len(routes)):
                if routes[j]['first'] == -1:
                    continue
                A = routes[i]
                B = routes[j]
                # Compute other_max
                other_max = 0.0
                for k, r in enumerate(routes):
                    if k != i and k != j:
                        other_max = max(other_max, r['total'])
                # Try 4 merges
                # 1. A then B (no reversal)
                new_total1 = A['total'] - A['distlast0'] + B['total'] - B['dist0first'] + dist[A['last']][B['first']]
                new_first1 = A['first']
                new_last1 = B['last']
                new_dist0first1 = A['dist0first']
                new_distlast0_1 = B['distlast0']
                cand_max1 = max(other_max, new_total1)
                # 2. A then rev B
                new_total2 = A['total'] - A['distlast0'] + B['total'] - B['distlast0'] + dist[A['last']][B['last']]
                new_first2 = A['first']
                new_last2 = B['first']
                new_dist0first2 = A['dist0first']
                new_distlast0_2 = B['dist0first']
                cand_max2 = max(other_max, new_total2)
                # 3. rev A then B
                new_total3 = A['total'] - A['dist0first'] + B['total'] - B['dist0first'] + dist[A['first']][B['first']]
                new_first3 = A['last']
                new_last3 = B['last']
                new_dist0first3 = A['distlast0']
                new_distlast0_3 = B['distlast0']
                cand_max3 = max(other_max, new_total3)
                # 4. rev A then rev B
                new_total4 = A['total'] - A['dist0first'] + B['total'] - B['distlast0'] + dist[A['first']][B['last']]
                new_first4 = A['last']
                new_last4 = B['first']
                new_dist0first4 = A['distlast0']
                new_distlast0_4 = B['dist0first']
                cand_max4 = max(other_max, new_total4)
                # Evaluate best max among these
                candidates = [
                    (cand_max1, i, j, 1, new_total1, new_first1, new_last1, new_dist0first1, new_distlast0_1),
                    (cand_max2, i, j, 2, new_total2, new_first2, new_last2, new_dist0first2, new_distlast0_2),
                    (cand_max3, i, j, 3, new_total3, new_first3, new_last3, new_dist0first3, new_distlast0_3),
                    (cand_max4, i, j, 4, new_total4, new_first4, new_last4, new_dist0first4, new_distlast0_4)
                ]
                for cand in candidates:
                    cand_max_val = cand[0]
                    if cand_max_val < best_max:
                        best_max = cand_max_val
                        best_i = i
                        best_j = j
                        # Store merge type and details
                        best_merge = (cand[3], cand[4], cand[5], cand[6], cand[7], cand[8])
                    elif cand_max_val == best_max:
                        # Tie-breaking: prefer smaller i, then j, then merge type
                        if i < best_i or (i == best_i and j < best_j) or (i == best_i and j == best_j and cand[3] < best_merge[0]):
                            best_max = cand_max_val
                            best_i = i
                            best_j = j
                            best_merge = (cand[3], cand[4], cand[5], cand[6], cand[7], cand[8])
        if best_i == -1:
            break  # no feasible merge (shouldn't happen)
        # Apply best merge
        i, j = best_i, best_j
        merge_type = best_merge[0]
        new_total = best_merge[1]
        new_first = best_merge[2]
        new_last = best_merge[3]
        new_dist0first = best_merge[4]
        new_distlast0 = best_merge[5]
        # Construct the merged route list
        A = routes[i]
        B = routes[j]
        if merge_type == 1:  # A then B
            new_route = A['route'] + B['route'][1:]
        elif merge_type == 2:  # A then rev B
            rev_B = [0] + B['route'][1:-1][::-1] + [0]
            new_route = A['route'] + rev_B[1:]
        elif merge_type == 3:  # rev A then B
            rev_A = [0] + A['route'][1:-1][::-1] + [0]
            new_route = rev_A + B['route'][1:]
        else:  # rev A then rev B
            rev_A = [0] + A['route'][1:-1][::-1] + [0]
            rev_B = [0] + B['route'][1:-1][::-1] + [0]
            new_route = rev_A + rev_B[1:]
        # Create new route dict
        new_route_dict = {
            'route': new_route,
            'first': new_first,
            'last': new_last,
            'dist0first': new_dist0first,
            'distlast0': new_distlast0,
            'total': new_total
        }
        # Remove old routes - pop larger index first
        if i > j:
            i, j = j, i  # ensure i < j for popping order
        routes.pop(j)
        routes.pop(i)
        routes.append(new_route_dict)
    
    # Now we have exactly truck_count routes (some may be empty)
    # Local search: relocate and swap to reduce max distance
    # We'll run a limited number of iterations
    n_cust = n - 1
    max_iter = max(10, n_cust * 2)
    best_max = current_max()
    best_routes = [r['route'].copy() for r in routes]
    
    for _ in range(max_iter):
        improved = False
        # Relocate: move a customer from one route to another
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                if len(routes[i]['route']) <= 2:  # only depot
                    continue
                # Try each customer in route i (excluding depots)
                route_i = routes[i]['route']
                route_j = routes[j]['route']
                # For each position in route_i (1 to len-2)
                for pos_i in range(1, len(route_i)-1):
                    customer = route_i[pos_i]
                    # For each insertion position in route_j (after depot 0, before last 0)
                    for pos_j in range(1, len(route_j)):  # after 0 up to last 0 (inclusive? we can insert before last 0)
                        # Compute new routes if we remove customer from i and insert at pos_j in j
                        new_route_i = route_i[:pos_i] + route_i[pos_i+1:]
                        new_route_j = route_j[:pos_j] + [customer] + route_j[pos_j:]
                        # Compute distances
                        def route_dist(r):
                            d = 0.0
                            for k in range(len(r)-1):
                                d += dist[r[k]][r[k+1]]
                            return d
                        new_dist_i = route_dist(new_route_i)
                        new_dist_j = route_dist(new_route_j)
                        # Compute new max
                        new_max = max(new_dist_i, new_dist_j)
                        for k, r in enumerate(routes):
                            if k != i and k != j:
                                new_max = max(new_max, r['total'])
                        if new_max < best_max:
                            # Accept move
                            best_max = new_max
                            # Update routes
                            routes[i] = {'route': new_route_i, 'first': new_route_i[1] if len(new_route_i)>2 else -1, 'last': new_route_i[-2] if len(new_route_i)>2 else -1, 'dist0first': dist[0][new_route_i[1]] if len(new_route_i)>2 else 0, 'distlast0': dist[new_route_i[-2]][0] if len(new_route_i)>2 else 0, 'total': new_dist_i}
                            routes[j] = {'route': new_route_j, 'first': new_route_j[1] if len(new_route_j)>2 else -1, 'last': new_route_j[-2] if len(new_route_j)>2 else -1, 'dist0first': dist[0][new_route_j[1]] if len(new_route_j)>2 else 0, 'distlast0': dist[new_route_j[-2]][0] if len(new_route_j)>2 else 0, 'total': new_dist_j}
                            # Update empty route handling (if a route becomes empty, it should be [0,0])
                            for k in range(len(routes)):
                                if len(routes[k]['route']) == 2 and routes[k]['route'][0]==0 and routes[k]['route'][1]==0:
                                    routes[k]['first'] = -1
                                    routes[k]['last'] = -1
                                    routes[k]['dist0first'] = 0.0
                                    routes[k]['distlast0'] = 0.0
                                    routes[k]['total'] = 0.0
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Swap customers between routes
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    if len(routes[i]['route']) <= 2 or len(routes[j]['route']) <= 2:
                        continue
                    route_i = routes[i]['route']
                    route_j = routes[j]['route']
                    for pos_i in range(1, len(route_i)-1):
                        for pos_j in range(1, len(route_j)-1):
                            cust_i = route_i[pos_i]
                            cust_j = route_j[pos_j]
                            new_route_i = route_i[:pos_i] + [cust_j] + route_i[pos_i+1:]
                            new_route_j = route_j[:pos_j] + [cust_i] + route_j[pos_j+1:]
                            def route_dist(r):
                                d = 0.0
                                for k in range(len(r)-1):
                                    d += dist[r[k]][r[k+1]]
                                return d
                            new_dist_i = route_dist(new_route_i)
                            new_dist_j = route_dist(new_route_j)
                            new_max = max(new_dist_i, new_dist_j)
                            for k, r in enumerate(routes):
                                if k != i and k != j:
                                    new_max = max(new_max, r['total'])
                            if new_max < best_max:
                                best_max = new_max
                                routes[i] = {'route': new_route_i, 'first': new_route_i[1] if len(new_route_i)>2 else -1, 'last': new_route_i[-2] if len(new_route_i)>2 else -1, 'dist0first': dist[0][new_route_i[1]] if len(new_route_i)>2 else 0, 'distlast0': dist[new_route_i[-2]][0] if len(new_route_i)>2 else 0, 'total': new_dist_i}
                                routes[j] = {'route': new_route_j, 'first': new_route_j[1] if len(new_route_j)>2 else -1, 'last': new_route_j[-2] if len(new_route_j)>2 else -1, 'dist0first': dist[0][new_route_j[1]] if len(new_route_j)>2 else 0, 'distlast0': dist[new_route_j[-2]][0] if len(new_route_j)>2 else 0, 'total': new_dist_j}
                                # Handle empty routes
                                for k in range(len(routes)):
                                    if len(routes[k]['route']) == 2 and routes[k]['route'][0]==0 and routes[k]['route'][1]==0:
                                        routes[k]['first'] = -1
                                        routes[k]['last'] = -1
                                        routes[k]['dist0first'] = 0.0
                                        routes[k]['distlast0'] = 0.0
                                        routes[k]['total'] = 0.0
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    # After local search, convert to output format
    # Ensure exactly truck_count routes
    # If we have fewer (should not happen), add empty routes
    while len(routes) < truck_count:
        routes.append({'route': [0,0], 'first': -1, 'last': -1, 'dist0first': 0.0, 'distlast0': 0.0, 'total': 0.0})
    # Ensure each route starts and ends with 0
    for r in routes:
        if r['route'][0] != 0:
            r['route'].insert(0, 0)
        if r['route'][-1] != 0:
            r['route'].append(0)
    # Return list of route lists
    result = [r['route'] for r in routes]
    return result