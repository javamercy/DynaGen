import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    num_customers = n - 1
    if truck_count < 1:
        return []
    if num_customers == 0:
        routes = [[0, 0] for _ in range(truck_count)]
        return routes

    def route_dist(r):
        d = 0
        for a, b in zip(r, r[1:]):
            d += distance_matrix[a][b]
        return d

    # Balanced Clarke-Wright savings construction
    customers = list(range(1, n))
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))

    # Initialize each customer as a separate route [0, c, 0]
    routes = [[0, c, 0] for c in customers]
    # Add empty routes to reach truck_count
    while len(routes) < truck_count:
        routes.append([0, 0])

    # Helper to get first and last customer
    def get_ends(r):
        if len(r) <= 2:
            return None, None
        return r[1], r[-2]

    cust_to_route = {}
    for idx, r in enumerate(routes):
        first, last = get_ends(r)
        if first is not None:
            cust_to_route[first] = idx
            cust_to_route[last] = idx

    # Merge savings until exactly truck_count routes
    idx = 0
    while len(routes) > truck_count and idx < len(savings):
        s, i, j = savings[idx]
        idx += 1
        if i not in cust_to_route or j not in cust_to_route:
            continue
        ri = cust_to_route[i]
        rj = cust_to_route[j]
        if ri == rj:
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        first_i, last_i = get_ends(route_i)
        first_j, last_j = get_ends(route_j)
        if last_i == i and first_j == j:
            # i at end of ri, j at start of rj
            new_route = route_i[:-1] + route_j[1:]
        elif first_i == i and last_j == j:
            new_route = route_j[:-1] + route_i[1:]
        else:
            continue
        # Balance: only accept merge if it does not create a too long route? We'll simply merge.
        routes[ri] = new_route
        routes.pop(rj)
        # Update mapping
        cust_to_route = {}
        for idx_r, r in enumerate(routes):
            first, last = get_ends(r)
            if first is not None:
                cust_to_route[first] = idx_r
                cust_to_route[last] = idx_r

    while len(routes) < truck_count:
        routes.append([0, 0])

    # Post-construction balancing: move customers from longest to shortest
    for _ in range(num_customers // truck_count):
        longest_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        shortest_idx = min(range(len(routes)), key=lambda i: route_dist(routes[i]))
        if longest_idx == shortest_idx:
            break
        best_max = route_dist(routes[longest_idx])
        best_move = None
        for pos, cust in enumerate(routes[longest_idx][1:-1]):
            new_long = routes[longest_idx][:pos+1] + routes[longest_idx][pos+2:]
            for ins in range(1, len(routes[shortest_idx])):
                new_short = routes[shortest_idx][:ins] + [cust] + routes[shortest_idx][ins:]
                new_max = max(route_dist(new_long), route_dist(new_short))
                if new_max < best_max:
                    best_max = new_max
                    best_move = (longest_idx, shortest_idx, pos, ins, cust)
        if best_move:
            li, si, pos, ins, cust = best_move
            routes[li] = routes[li][:pos+1] + routes[li][pos+2:]
            routes[si] = routes[si][:ins] + [cust] + routes[si][ins:]

    report_best_vrp(routes)

    # Threshold accepting local search
    max_iter_ta = max(20, num_customers * 2)
    threshold = 0.1 * max(route_dist(r) for r in routes)  # 10% of current max
    for _ in range(max_iter_ta):
        improved = False
        for li in range(len(routes)):
            for pos, cust in enumerate(routes[li][1:-1]):
                # Try move to other routes
                for other in range(len(routes)):
                    if other == li:
                        continue
                    new_route_li = routes[li][:pos+1] + routes[li][pos+2:]
                    for ins in range(1, len(routes[other])):
                        new_route_other = routes[other][:ins] + [cust] + routes[other][ins:]
                        new_max = max(route_dist(new_route_li), route_dist(new_route_other), max(route_dist(r) for r in [routes[i] for i in range(len(routes)) if i not in (li, other)]))
                        old_max = max(route_dist(routes[li]), route_dist(routes[other]), max(route_dist(r) for r in [routes[i] for i in range(len(routes)) if i not in (li, other)]))
                        if new_max < old_max + threshold:
                            # Accept if within threshold
                            routes[li] = new_route_li
                            routes[other] = new_route_other
                            improved = True
                            report_best_vrp(routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            threshold *= 0.9  # reduce threshold
            if threshold < 1e-6:
                break

    # Intra-route 2-opt on the longest route
    for _ in range(max(10, num_customers // 2)):
        longest_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        route = routes[longest_idx]
        best_imp = 0
        best_i = best_j = -1
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                old = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                new = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                if new < old - best_imp:
                    best_imp = old - new
                    best_i = i
                    best_j = j
        if best_imp > 0:
            routes[longest_idx] = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
            report_best_vrp(routes)
        else:
            break

    # Restarts with perturbation
    best_routes = [r[:] for r in routes]
    best_max = max(route_dist(r) for r in routes)
    for restart in range(max(3, num_customers // 20)):
        # Perturb: remove a few customers from longest routes and reinsert greedily
        current_routes = [r[:] for r in best_routes]
        # Identify customers to remove (e.g., from the longest route)
        longest_idx = max(range(len(current_routes)), key=lambda i: route_dist(current_routes[i]))
        route = current_routes[longest_idx]
        if len(route) <= 3:
            break
        # Remove a set of customers (e.g., every other customer from longest route)
        removed = []
        for pos in range(len(route)-2, 0, -1):
            if len(route) <= 3:
                break
            cust = route.pop(pos)
            removed.append(cust)
            if len(removed) >= max(2, num_customers // 10):
                break
        # Greedy reinsert: for each removed customer, insert in best position to minimize max
        for cust in removed:
            best_max_local = float('inf')
            best_route_idx = -1
            best_pos = -1
            for ri, r in enumerate(current_routes):
                for pos in range(1, len(r)):
                    new_r = r[:pos] + [cust] + r[pos:]
                    new_max_candidate = max(route_dist(new_r), max(route_dist(r) for r in [current_routes[i] for i in range(len(current_routes)) if i != ri]))
                    if new_max_candidate < best_max_local:
                        best_max_local = new_max_candidate
                        best_route_idx = ri
                        best_pos = pos
            if best_route_idx != -1:
                current_routes[best_route_idx] = current_routes[best_route_idx][:best_pos] + [cust] + current_routes[best_route_idx][best_pos:]
        # Apply local search again (simplified: just a few moves)
        for _ in range(10):
            li = max(range(len(current_routes)), key=lambda i: route_dist(current_routes[i]))
            for pos, cust in enumerate(current_routes[li][1:-1]):
                for other in range(len(current_routes)):
                    if other == li:
                        continue
                    new_li = current_routes[li][:pos+1] + current_routes[li][pos+2:]
                    for ins in range(1, len(current_routes[other])):
                        new_other = current_routes[other][:ins] + [cust] + current_routes[other][ins:]
                        new_max = max(route_dist(new_li), route_dist(new_other), max(route_dist(r) for r in [current_routes[i] for i in range(len(current_routes)) if i not in (li, other)]))
                        if new_max < best_max:
                            current_routes[li] = new_li
                            current_routes[other] = new_other
                            best_max = new_max
                            report_best_vrp(current_routes)
                            break
                    if best_max < best_max:
                        break
                if best_max < best_max:
                    break
        # Update best
        current_max = max(route_dist(r) for r in current_routes)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in current_routes]

    return best_routes