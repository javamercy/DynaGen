import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    num_customers = n - 1
    if truck_count < 1:
        return []

    customers = list(range(1, n))
    routes = [[0, c, 0] for c in customers]
    while len(routes) < truck_count:
        routes.append([0, 0])

    if num_customers == 0:
        while len(routes) < truck_count:
            routes.append([0, 0])
        routes = routes[:truck_count]
        return routes

    # Compute savings
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))

    cust_to_route = {}
    route_first = {}
    route_last = {}
    for idx, route in enumerate(routes):
        if len(route) > 2:
            first = route[1]
            last = route[-2]
            route_first[idx] = first
            route_last[idx] = last
            cust_to_route[first] = idx
            cust_to_route[last] = idx
        else:
            route_first[idx] = None
            route_last[idx] = None

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
        i_first = (route_i[1] == i)
        i_last = (route_i[-2] == i)
        j_first = (route_j[1] == j)
        j_last = (route_j[-2] == j)
        if not ((i_first or i_last) and (j_first or j_last)):
            continue
        if i_last and j_first:
            new_route = route_i[:-1] + route_j[1:]
        elif i_first and j_last:
            new_route = route_j[:-1] + route_i[1:]
        else:
            continue
        new_routes = []
        for idx_r, r in enumerate(routes):
            if idx_r != ri and idx_r != rj:
                new_routes.append(r)
        new_routes.append(new_route)
        routes = new_routes
        cust_to_route = {}
        route_first = {}
        route_last = {}
        for idx_r, r in enumerate(routes):
            if len(r) > 2:
                first = r[1]
                last = r[-2]
                route_first[idx_r] = first
                route_last[idx_r] = last
                cust_to_route[first] = idx_r
                cust_to_route[last] = idx_r
            else:
                route_first[idx_r] = None
                route_last[idx_r] = None

    while len(routes) < truck_count:
        routes.append([0, 0])

    def route_dist(r):
        d = 0
        for a, b in zip(r, r[1:]):
            d += distance_matrix[a][b]
        return d

    # Report initial solution
    report_best_vrp(routes)

    # Balancing: move customers from longest to shortest route
    max_iter = num_customers
    for _ in range(max_iter):
        longest_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        shortest_idx = min(range(len(routes)), key=lambda i: route_dist(routes[i]))
        if shortest_idx == longest_idx:
            break
        best_max = route_dist(routes[longest_idx])
        best_move = None
        longest_route = routes[longest_idx]
        shortest_route = routes[shortest_idx]
        for pos, cust in enumerate(longest_route[1:-1]):
            new_longest = longest_route[:pos+1] + longest_route[pos+2:]
            for ins_pos in range(1, len(shortest_route)):
                new_shortest = shortest_route[:ins_pos] + [cust] + shortest_route[ins_pos:]
                new_max = max(route_dist(new_longest), route_dist(new_shortest))
                if new_max < best_max:
                    best_max = new_max
                    best_move = (pos, ins_pos, cust)
        if best_move is not None:
            pos, ins_pos, cust = best_move
            routes[longest_idx] = longest_route[:pos+1] + longest_route[pos+2:]
            routes[shortest_idx] = shortest_route[:ins_pos] + [cust] + shortest_route[ins_pos:]
            report_best_vrp(routes)
        else:
            break

    return routes