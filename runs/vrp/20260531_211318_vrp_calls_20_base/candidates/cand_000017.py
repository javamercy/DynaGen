import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    dist = distance_matrix

    def route_distance(route):
        if not route:
            return 0.0
        d = dist[0, route[0]] + dist[route[-1], 0]
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    def insertion_cost(route, cust, pos, curr_dist):
        if not route:
            return dist[0, cust] + dist[cust, 0]
        if pos == 0:
            return curr_dist - dist[0, route[0]] + dist[0, cust] + dist[cust, route[0]]
        elif pos == len(route):
            return curr_dist - dist[route[-1], 0] + dist[route[-1], cust] + dist[cust, 0]
        else:
            prev = route[pos-1]
            nxt = route[pos]
            return curr_dist - dist[prev, nxt] + dist[prev, cust] + dist[cust, nxt]

    def removal_cost(route, i, curr_dist):
        if len(route) == 1:
            return 0.0
        if i == 0:
            return curr_dist - dist[0, route[0]] - dist[route[0], route[1]] + dist[0, route[1]]
        elif i == len(route)-1:
            return curr_dist - dist[route[-2], route[-1]] - dist[route[-1], 0] + dist[route[-2], 0]
        else:
            prev = route[i-1]
            nxt = route[i+1]
            return curr_dist - dist[prev, route[i]] - dist[route[i], nxt] + dist[prev, nxt]

    def max_other(route_dists, *exclude):
        best = 0.0
        for t, d in enumerate(route_dists):
            if t not in exclude:
                best = max(best, d)
        return best

    # Construction
    routes = [[] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    for cust in customers:
        best_truck = None
        best_pos = None
        best_new_max = float('inf')
        best_new_dist = float('inf')
        for t in range(truck_count):
            route = routes[t]
            curr_dist = route_dists[t]
            for pos in range(len(route)+1):
                new_dist = insertion_cost(route, cust, pos, curr_dist)
                other_max = max_other(route_dists, t)
                new_max = max(new_dist, other_max)
                if (new_max, new_dist) < (best_new_max, best_new_dist):
                    best_new_max = new_max
                    best_new_dist = new_dist
                    best_truck = t
                    best_pos = pos
        route = routes[best_truck]
        route.insert(best_pos, cust)
        route_dists[best_truck] = best_new_dist

    full_routes = [[0] + r + [0] for r in routes]
    try:
        report_best_vrp(full_routes)
    except NameError:
        pass
    max_dist = max(route_dists)

    # Improvement
    max_passes = 10 * n * truck_count
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # Relocate: move one customer to another route
        for t_from in range(truck_count):
            if len(routes[t_from]) == 0:
                continue
            for i in range(len(routes[t_from])):
                cust = routes[t_from][i]
                old_dist_from = route_dists[t_from]
                new_dist_from = removal_cost(routes[t_from], i, old_dist_from)
                for t_to in range(truck_count):
                    if t_to == t_from:
                        continue
                    route_to = routes[t_to]
                    old_dist_to = route_dists[t_to]
                    for j in range(len(route_to)+1):
                        new_dist_to = insertion_cost(route_to, cust, j, old_dist_to)
                        other_max = max_other(route_dists, t_from, t_to)
                        new_max = max(new_dist_from, new_dist_to, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t_from].pop(i)
                            route_dists[t_from] = new_dist_from
                            routes[t_to].insert(j, cust)
                            route_dists[t_to] = new_dist_to
                            max_dist = new_max
                            improved = True
                            full_routes = [[0] + r + [0] for r in routes]
                            try:
                                report_best_vrp(full_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Swap: exchange two customers from different routes
        for t1 in range(truck_count):
            if len(routes[t1]) == 0:
                continue
            for i in range(len(routes[t1])):
                cust1 = routes[t1][i]
                for t2 in range(t1+1, truck_count):
                    if len(routes[t2]) == 0:
                        continue
                    for j in range(len(routes[t2])):
                        cust2 = routes[t2][j]
                        new_route1 = routes[t1].copy()
                        new_route2 = routes[t2].copy()
                        new_route1[i] = cust2
                        new_route2[j] = cust1
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        other_max = max_other(route_dists, t1, t2)
                        new_max = max(new_dist1, new_dist2, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            route_dists[t1] = new_dist1
                            route_dists[t2] = new_dist2
                            max_dist = new_max
                            improved = True
                            full_routes = [[0] + r + [0] for r in routes]
                            try:
                                report_best_vrp(full_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Intra-route 2-opt
        for t in range(truck_count):
            route = routes[t]
            if len(route) < 2:
                continue
            for i in range(len(route)-1):
                for j in range(i+1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < route_dists[t] - 1e-9:
                        other_max = max_other(route_dists, t)
                        new_max = max(new_dist, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t] = new_route
                            route_dists[t] = new_dist
                            max_dist = new_max
                            improved = True
                            full_routes = [[0] + r + [0] for r in routes]
                            try:
                                report_best_vrp(full_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Cross-route exchange (2-opt*)
        for t1 in range(truck_count):
            for t2 in range(t1+1, truck_count):
                route1 = routes[t1]
                route2 = routes[t2]
                if not route1 or not route2:
                    continue
                for i in range(len(route1)+1):
                    for j in range(len(route2)+1):
                        new_route1 = route1[:i] + route2[j:]
                        new_route2 = route2[:j] + route1[i:]
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        other_max = max_other(route_dists, t1, t2)
                        new_max = max(new_dist1, new_dist2, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            route_dists[t1] = new_dist1
                            route_dists[t2] = new_dist2
                            max_dist = new_max
                            improved = True
                            full_routes = [[0] + r + [0] for r in routes]
                            try:
                                report_best_vrp(full_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    final_routes = [[0] + r + [0] for r in routes]
    return final_routes