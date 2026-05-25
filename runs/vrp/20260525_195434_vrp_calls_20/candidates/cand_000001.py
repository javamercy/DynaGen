import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    d = distance_matrix

    def route_distance(route):
        return sum(d[route[i]][route[i+1]] for i in range(len(route)-1))

    # Initialize each customer as a separate route
    routes = [[0, i, 0] for i in range(1, n)]
    if len(routes) <= truck_count:
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # Build savings list
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s = d[0][i] + d[0][j] - d[i][j]
            savings.append((-s, i, j))
    savings.sort()

    # Node to route mapping
    node_to_route = {}
    for idx, r in enumerate(routes):
        for node in r[1:-1]:
            node_to_route[node] = idx

    route_dists = [route_distance(r) for r in routes]

    # Apply savings merges
    for _, i, j in savings:
        if len(routes) <= truck_count:
            break
        ri = node_to_route.get(i)
        rj = node_to_route.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        ends_i = (route_i[1], route_i[-2])
        ends_j = (route_j[1], route_j[-2])
        if i not in ends_i or j not in ends_j:
            continue
        # Perform merge
        cust_i = route_i[1:-1]
        if i == route_i[1]:
            cust_i = cust_i[::-1]
        cust_j = route_j[1:-1]
        if j == route_j[-2]:
            cust_j = cust_j[::-1]
        new_cust = cust_i + cust_j
        new_route = [0] + new_cust + [0]
        # Remove old routes and add new
        if ri > rj:
            routes.pop(ri)
            routes.pop(rj)
        else:
            routes.pop(rj)
            routes.pop(ri)
        routes.append(new_route)
        route_dists.append(route_distance(new_route))
        # Update mapping
        for node in route_i[1:-1]:
            del node_to_route[node]
        for node in route_j[1:-1]:
            del node_to_route[node]
        for node in new_cust:
            node_to_route[node] = len(routes)-1

    # Force merge to reach truck_count
    while len(routes) > truck_count:
        best_pair = None
        best_savings = -float('inf')
        best_new_route = None
        for a in range(len(routes)):
            for b in range(a+1, len(routes)):
                ra = routes[a]
                rb = routes[b]
                if len(ra) <= 2 or len(rb) <= 2:
                    continue
                ends_a = [ra[1], ra[-2]]
                ends_b = [rb[1], rb[-2]]
                for i in ends_a:
                    for j in ends_b:
                        cust_a = ra[1:-1]
                        if i == ra[1]:
                            cust_a = cust_a[::-1]
                        cust_b = rb[1:-1]
                        if j == rb[-2]:
                            cust_b = cust_b[::-1]
                        new_cust = cust_a + cust_b
                        new_route = [0] + new_cust + [0]
                        new_dist = route_distance(new_route)
                        current_sum = route_dists[a] + route_dists[b]
                        savings_val = current_sum - new_dist
                        if savings_val > best_savings:
                            best_savings = savings_val
                            best_pair = (a, b, i, j)
                            best_new_route = new_route
        if best_pair is None:
            break
        a, b, i, j = best_pair
        if a > b:
            routes.pop(a)
            routes.pop(b)
        else:
            routes.pop(b)
            routes.pop(a)
        routes.append(best_new_route)
        route_dists = [route_distance(r) for r in routes]
        node_to_route.clear()
        for idx, r in enumerate(routes):
            for node in r[1:-1]:
                node_to_route[node] = idx

    while len(routes) < truck_count:
        routes.append([0, 0])

    report_best_vrp(routes)

    # Improvement phase
    max_iter = 10 * n
    for _ in range(max_iter):
        route_dists = [route_distance(r) for r in routes]
        current_max = max(route_dists)
        longest_indices = [i for i, d in enumerate(route_dists) if d == current_max]
        improved = False
        for idx in longest_indices:
            route = routes[idx]
            if len(route) <= 2:
                continue
            customers = route[1:-1]
            for cust in customers:
                new_route_no_cust = [0] + [node for node in route[1:-1] if node != cust] + [0]
                for other_idx in range(len(routes)):
                    if other_idx == idx or len(routes[other_idx]) <= 2:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        new_other = other_route[:pos] + [cust] + other_route[pos:]
                        dist_no_cust = route_distance(new_route_no_cust)
                        dist_new_other = route_distance(new_other)
                        new_route_dists = route_dists.copy()
                        new_route_dists[idx] = dist_no_cust
                        new_route_dists[other_idx] = dist_new_other
                        new_max = max(new_route_dists)
                        if new_max < current_max:
                            routes[idx] = new_route_no_cust
                            routes[other_idx] = new_other
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
            for idx in longest_indices:
                route = routes[idx]
                if len(route) <= 3:
                    continue
                best_gain = 0
                best_i = best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+2, len(route)-1):
                        gain = d[route[i]][route[i+1]] + d[route[j]][route[j+1]] - d[route[i]][route[j]] - d[route[i+1]][route[j+1]]
                        if gain > best_gain:
                            best_gain = gain
                            best_i, best_j = i, j
                if best_gain > 1e-9:
                    route[best_i+1:best_j+1] = route[best_i+1:best_j+1][::-1]
                    new_dist = route_distance(route)
                    route_dists[idx] = new_dist
                    if new_dist < current_max:
                        report_best_vrp(routes)
                    improved = True
                    break
        if not improved:
            break
    return routes