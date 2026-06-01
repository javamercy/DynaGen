import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = len(distance_matrix)
    customers = list(range(1, n))
    best_routes = None
    best_max_dist = float('inf')

    def route_distance(route):
        if len(route) <= 1:
            return 0
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    def report_best_vrp(routes):
        nonlocal best_routes, best_max_dist
        max_d = max(route_distance(r) for r in routes)
        if max_d < best_max_dist:
            best_max_dist = max_d
            best_routes = [list(r) for r in routes]

    # Savings initialization (Clarke-Wright)
    # Each customer starts as its own route
    routes = [[0, c, 0] for c in customers]
    if len(routes) <= truck_count:
        # pad with empty routes
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return best_routes

    # Compute savings for all pairs
    savings_list = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings_list.append((s, i, j))
    savings_list.sort(reverse=True, key=lambda x: x[0])

    # Function to find route containing a customer
    def find_route(cust):
        for idx, r in enumerate(routes):
            if cust in r:
                return idx
        return None

    # Merge routes until we have truck_count routes
    while len(routes) > truck_count:
        best_sav = float('-inf')
        best_pair = None
        for s, i, j in savings_list:
            if s <= best_sav:
                break
            idx_i = find_route(i)
            idx_j = find_route(j)
            if idx_i is None or idx_j is None or idx_i == idx_j:
                continue
            route_i = routes[idx_i]
            route_j = routes[idx_j]
            # Check eligibility: i is last customer of route_i and j is first of route_j
            if (route_i[-2] == i and route_j[1] == j):
                best_sav = s
                best_pair = (idx_i, idx_j, 'i_last_j_first')
                break
            # Check eligibility: i is first of route_i and j is last of route_j
            elif (route_i[1] == i and route_j[-2] == j):
                best_sav = s
                best_pair = (idx_i, idx_j, 'i_first_j_last')
                break
        if best_pair is None:
            # Fallback: merge two shortest routes (by number of customers)
            # Sort routes by length (number of customers)
            route_indices_sorted = sorted(range(len(routes)), key=lambda idx: len(routes[idx])-2)  # -2 for depot
            idx_i = route_indices_sorted[0]
            idx_j = route_indices_sorted[1]
            if idx_i == idx_j:
                # Only one route left? Should not happen because len(routes) > truck_count >=1
                break
            # Merge by concatenating (route_i[:-1] + route_j[1:])
            new_route = routes[idx_i][:-1] + routes[idx_j][1:]
            # Remove both and add new
            if idx_i < idx_j:
                routes.pop(idx_j)
                routes.pop(idx_i)
            else:
                routes.pop(idx_i)
                routes.pop(idx_j)
            routes.append(new_route)
            continue
        idx_i, idx_j, merge_type = best_pair
        route_i = routes[idx_i]
        route_j = routes[idx_j]
        if merge_type == 'i_last_j_first':
            new_route = route_i[:-1] + route_j[1:]
        else:  # i_first_j_last
            new_route = route_j[:-1] + route_i[1:]
        # Remove both routes and add new
        if idx_i < idx_j:
            routes.pop(idx_j)
            routes.pop(idx_i)
        else:
            routes.pop(idx_i)
            routes.pop(idx_j)
        routes.append(new_route)

    # Pad with empty routes if fewer than truck_count
    while len(routes) < truck_count:
        routes.append([0, 0])

    report_best_vrp(routes)

    # Improvement: local search with relocate, swap, and intra-route Or-opt
    max_iter = min(n * truck_count, 200)
    for iteration in range(max_iter):
        improved = False
        # inter-route relocate
        for i in range(truck_count):
            route_i = routes[i]
            if len(route_i) <= 2:
                continue
            for cust in route_i[1:-1]:
                for j in range(truck_count):
                    if i == j:
                        continue
                    route_j = routes[j]
                    for pos in range(1, len(route_j)):
                        new_routes = [list(r) for r in routes]
                        new_routes[i].remove(cust)
                        new_routes[j].insert(pos, cust)
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max_dist:
                            routes = new_routes
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # inter-route swap
        for i in range(truck_count):
            route_i = routes[i]
            if len(route_i) <= 2:
                continue
            for cust_i in route_i[1:-1]:
                for j in range(i+1, truck_count):
                    route_j = routes[j]
                    if len(route_j) <= 2:
                        continue
                    for cust_j in route_j[1:-1]:
                        new_routes = [list(r) for r in routes]
                        idx_i = new_routes[i].index(cust_i)
                        idx_j = new_routes[j].index(cust_j)
                        new_routes[i][idx_i], new_routes[j][idx_j] = cust_j, cust_i
                        new_max = max(route_distance(r) for r in new_routes)
                        if new_max < best_max_dist:
                            routes = new_routes
                            report_best_vrp(routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # intra-route Or-opt
        for i in range(truck_count):
            route = routes[i]
            if len(route) <= 3:
                continue
            best_route = list(route)
            best_dist = route_distance(route)
            for seg_len in range(1, min(4, len(route)-1)):
                for start in range(1, len(route)-seg_len):
                    end = start + seg_len - 1
                    segment = route[start:end+1]
                    remaining = route[:start] + route[end+1:]
                    for pos in range(1, len(remaining)):
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        if new_route[0] != 0 or new_route[-1] != 0:
                            continue
                        d = route_distance(new_route)
                        if d < best_dist:
                            best_dist = d
                            best_route = new_route
                            improved = True
            if improved:
                routes[i] = best_route
                report_best_vrp(routes)
        if not improved:
            break
    max_d = max(route_distance(r) for r in routes)
    if max_d < best_max_dist:
        best_routes = [list(r) for r in routes]
    return best_routes