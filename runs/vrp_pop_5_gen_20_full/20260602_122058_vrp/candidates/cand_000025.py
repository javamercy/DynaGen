import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    num_customers = n - 1
    if truck_count < 1:
        return []
    if num_customers == 0:
        return [[0,0] for _ in range(truck_count)]

    customers = list(range(1, n))
    # Initialize each customer as its own route
    routes = [[0, c, 0] for c in customers]
    while len(routes) < truck_count:
        routes.append([0, 0])

    # Compute savings
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                savings.append((s, i, j))
    savings.sort(key=lambda x: (-x[0], x[1], x[2]))

    # Data structures for merging
    cust_to_route = {}
    route_first = {}
    route_last = {}
    for idx, r in enumerate(routes):
        if len(r) > 2:
            first = r[1]
            last = r[-2]
            route_first[idx] = first
            route_last[idx] = last
            cust_to_route[first] = idx
            cust_to_route[last] = idx
        else:
            route_first[idx] = None
            route_last[idx] = None

    # Merge routes using savings until exactly truck_count routes
    savings_idx = 0
    while len(routes) > truck_count and savings_idx < len(savings):
        s, i, j = savings[savings_idx]
        savings_idx += 1
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
        # Remove the two old routes and add new one
        new_routes = []
        for idx_r, r in enumerate(routes):
            if idx_r != ri and idx_r != rj:
                new_routes.append(r)
        new_routes.append(new_route)
        routes = new_routes
        # Rebuild data structures
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

    # Ensure truck_count routes
    while len(routes) < truck_count:
        routes.append([0, 0])

    # Helper functions
    def route_dist(r):
        d = 0
        for a, b in zip(r, r[1:]):
            d += distance_matrix[a][b]
        return d

    def max_route_dist(routes):
        return max(route_dist(r) for r in routes)

    # Report initial solution
    report_best_vrp(routes)

    # Balancing: move customers from longest to shortest route
    for _ in range(num_customers):
        longest_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        shortest_idx = min(range(len(routes)), key=lambda i: route_dist(routes[i]))
        if shortest_idx == longest_idx:
            break
        best_max = max_route_dist(routes)
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
            routes[longest_idx] = routes[longest_idx][:pos+1] + routes[longest_idx][pos+2:]
            routes[shortest_idx] = routes[shortest_idx][:ins_pos] + [cust] + routes[shortest_idx][ins_pos:]
            report_best_vrp(routes)
        else:
            break

    # Local search
    n_cust = num_customers
    max_iter = n_cust * truck_count  # bounded
    for iteration in range(max_iter):
        improved = False
        current_max = max_route_dist(routes)
        # Move: remove customer from one route, insert into best position in another
        for ri in range(len(routes)):
            for rj in range(len(routes)):
                if ri == rj:
                    continue
                for pos_i, cust in enumerate(routes[ri][1:-1]):
                    new_ri = routes[ri][:pos_i+1] + routes[ri][pos_i+2:]
                    best_new_max = current_max
                    best_ins_pos = None
                    for ins_pos in range(1, len(routes[rj])):
                        new_rj = routes[rj][:ins_pos] + [cust] + routes[rj][ins_pos:]
                        # compute new max without recalculating all routes
                        # we can compute new_ri_dist, new_rj_dist, and max with others
                        new_ri_dist = route_dist(new_ri)
                        new_rj_dist = route_dist(new_rj)
                        new_max = max(new_ri_dist, new_rj_dist)
                        # check if other routes exceed
                        for idx, r in enumerate(routes):
                            if idx != ri and idx != rj:
                                d = route_dist(r)
                                if d > new_max:
                                    new_max = d
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_ins_pos = ins_pos
                    if best_ins_pos is not None and best_new_max < current_max:
                        # perform move
                        routes[ri] = new_ri
                        routes[rj] = routes[rj][:best_ins_pos] + [cust] + routes[rj][best_ins_pos:]
                        report_best_vrp(routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Swap customers between two routes
        for ri in range(len(routes)):
            for rj in range(ri+1, len(routes)):
                for pos_i, cust_i in enumerate(routes[ri][1:-1]):
                    for pos_j, cust_j in enumerate(routes[rj][1:-1]):
                        new_ri = routes[ri][:pos_i+1] + [cust_j] + routes[ri][pos_i+2:]
                        new_rj = routes[rj][:pos_j+1] + [cust_i] + routes[rj][pos_j+2:]
                        new_max = max(route_dist(new_ri), route_dist(new_rj))
                        for idx, r in enumerate(routes):
                            if idx != ri and idx != rj:
                                d = route_dist(r)
                                if d > new_max:
                                    new_max = d
                        if new_max < current_max:
                            routes[ri] = new_ri
                            routes[rj] = new_rj
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

        # 2-opt on the longest route
        longest_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        r = routes[longest_idx]
        best_improvement = 0
        best_pair = None
        for i in range(1, len(r)-2):
            for j in range(i+1, len(r)-1):
                # reverse segment from i to j
                new_r = r[:i] + r[i:j+1][::-1] + r[j+1:]
                new_dist = route_dist(new_r)
                old_dist = route_dist(r)
                if new_dist < old_dist - best_improvement + 1e-9:
                    best_improvement = old_dist - new_dist
                    best_pair = (i, j, new_r)
        if best_improvement > 0:
            routes[longest_idx] = best_pair[2]
            report_best_vrp(routes)
            improved = True

        if not improved:
            break

    return routes