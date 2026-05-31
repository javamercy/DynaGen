import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    dist = distance_matrix

    def route_distance(route):
        if not route:
            return 0.0
        d = dist[0, route[0]] + dist[route[-1], 0]
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    # Construction: sort customers by distance from depot
    customers = sorted(range(1, n), key=lambda i: dist[0, i])
    routes = [[] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count

    for cust in customers:
        best_truck = -1
        best_pos = -1
        best_max = float('inf')
        best_dist = float('inf')
        for t in range(truck_count):
            for pos in range(len(routes[t])+1):
                new_route = routes[t][:pos] + [cust] + routes[t][pos:]
                new_dist = route_distance(new_route)
                other_max = max(route_dists[:t] + route_dists[t+1:], default=0.0)
                new_max = max(new_dist, other_max)
                if new_max < best_max or (new_max == best_max and (t < best_truck or (t == best_truck and pos < best_pos))):
                    best_max = new_max
                    best_dist = new_dist
                    best_truck = t
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)
        route_dists[best_truck] = best_dist

    full_routes = [[0] + r + [0] for r in routes]
    report_best_vrp(full_routes)
    current_max = max(route_dists) if route_dists else 0.0

    # Improvement: best-improvement relocate, swap, and intra-route 2-opt
    max_passes = 10 * n * truck_count
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1

        # 2-opt on each route (best-improvement)
        for t in range(truck_count):
            if len(routes[t]) < 3:
                continue
            route = routes[t]
            best_route = route[:]
            best_dist = route_distance(best_route)
            while True:
                improved_2opt = False
                best_delta = 0.0
                best_i = best_j = -1
                for i in range(len(best_route)-1):
                    for j in range(i+1, len(best_route)):
                        new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                        new_dist = route_distance(new_route)
                        delta = new_dist - best_dist
                        if delta < -1e-12 and delta < best_delta:
                            best_delta = delta
                            best_route = new_route
                            best_dist = new_dist
                            improved_2opt = True
                            best_i, best_j = i, j
                if not improved_2opt:
                    break
                # apply best improvement found
            if best_dist < route_dists[t]:
                routes[t] = best_route[:]
                route_dists[t] = best_dist
                current_max = max(route_dists)
                full_routes = [[0] + r + [0] for r in routes]
                report_best_vrp(full_routes)
                improved = True

        # Relocate best-improvement
        best_relocate = None
        best_new_max = float('inf')
        best_tie = (truck_count, 0, 0, 0)  # (t_from, i, t_to, j)
        for t_from in range(truck_count):
            if len(routes[t_from]) == 0:
                continue
            for i in range(len(routes[t_from])):
                cust = routes[t_from][i]
                new_route_from = routes[t_from][:i] + routes[t_from][i+1:]
                new_dist_from = route_distance(new_route_from)
                for t_to in range(truck_count):
                    if t_to == t_from:
                        continue
                    for j in range(len(routes[t_to])+1):
                        new_route_to = routes[t_to][:j] + [cust] + routes[t_to][j:]
                        new_dist_to = route_distance(new_route_to)
                        other_max = max([route_dists[t] for t in range(truck_count) if t not in (t_from, t_to)], default=0.0)
                        new_max = max(new_dist_from, new_dist_to, other_max)
                        tie = (t_from, i, t_to, j)
                        if new_max < best_new_max or (new_max == best_new_max and tie < best_tie):
                            best_new_max = new_max
                            best_relocate = (t_from, i, t_to, j, new_route_from, new_route_to, new_dist_from, new_dist_to)
                            best_tie = tie
        if best_relocate and best_new_max < current_max:
            t_from, i, t_to, j, new_route_from, new_route_to, new_dist_from, new_dist_to = best_relocate
            routes[t_from] = new_route_from
            route_dists[t_from] = new_dist_from
            routes[t_to] = new_route_to
            route_dists[t_to] = new_dist_to
            current_max = best_new_max
            full_routes = [[0] + r + [0] for r in routes]
            report_best_vrp(full_routes)
            improved = True

        # Swap best-improvement
        best_swap = None
        best_new_max = float('inf')
        best_tie = (truck_count, 0, 0, 0)  # (t1, i, t2, j)
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
                        other_max = max([route_dists[t] for t in range(truck_count) if t not in (t1, t2)], default=0.0)
                        new_max = max(new_dist1, new_dist2, other_max)
                        tie = (t1, i, t2, j)
                        if new_max < best_new_max or (new_max == best_new_max and tie < best_tie):
                            best_new_max = new_max
                            best_swap = (t1, i, t2, j, new_route1, new_route2, new_dist1, new_dist2)
                            best_tie = tie
        if best_swap and best_new_max < current_max:
            t1, i, t2, j, new_route1, new_route2, new_dist1, new_dist2 = best_swap
            routes[t1] = new_route1
            route_dists[t1] = new_dist1
            routes[t2] = new_route2
            route_dists[t2] = new_dist2
            current_max = best_new_max
            full_routes = [[0] + r + [0] for r in routes]
            report_best_vrp(full_routes)
            improved = True

    final_routes = [[0] + r + [0] for r in routes]
    return final_routes