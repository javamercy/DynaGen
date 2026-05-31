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

    # Construction: sequential greedy insertion minimizing max route distance
    routes = [[] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    customers = list(range(1, n))
    for cust in customers:
        best_max = float('inf')
        best_total = float('inf')
        best_truck = None
        best_pos = None
        for t in range(truck_count):
            route = routes[t]
            curr_dist = route_dists[t]
            for pos in range(len(route)+1):
                new_route = route[:pos] + [cust] + route[pos:]
                new_dist = route_distance(new_route)
                other_max = max(route_dists[:t] + route_dists[t+1:])
                new_max = max(new_dist, other_max)
                new_total = new_dist + sum(route_dists[:t] + route_dists[t+1:])
                if (new_max, new_total) < (best_max, best_total):
                    best_max = new_max
                    best_total = new_total
                    best_truck = t
                    best_pos = pos
        routes[best_truck].insert(best_pos, cust)
        route_dists[best_truck] = route_distance(routes[best_truck])
        full_routes = [[0] + r + [0] for r in routes]
        report_best_vrp(full_routes)

    max_dist = max(route_dists)
    # Improvement loops
    max_passes = n * truck_count
    for _ in range(max_passes):
        improved = False
        # Relocate
        for t_from in range(truck_count):
            if not routes[t_from]:
                continue
            for i in range(len(routes[t_from])):
                cust = routes[t_from][i]
                for t_to in range(truck_count):
                    if t_to == t_from:
                        continue
                    for j in range(len(routes[t_to])+1):
                        new_route_from = routes[t_from][:i] + routes[t_from][i+1:]
                        new_dist_from = route_distance(new_route_from)
                        new_route_to = routes[t_to][:j] + [cust] + routes[t_to][j:]
                        new_dist_to = route_distance(new_route_to)
                        other_max = max(route_dists[t] for t in range(truck_count) if t != t_from and t != t_to)
                        new_max = max(new_dist_from, new_dist_to, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t_from] = new_route_from
                            routes[t_to] = new_route_to
                            route_dists[t_from] = new_dist_from
                            route_dists[t_to] = new_dist_to
                            max_dist = new_max
                            full_routes = [[0] + r + [0] for r in routes]
                            report_best_vrp(full_routes)
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
        # Swap
        for t1 in range(truck_count):
            if not routes[t1]:
                continue
            for i in range(len(routes[t1])):
                for t2 in range(t1+1, truck_count):
                    if not routes[t2]:
                        continue
                    for j in range(len(routes[t2])):
                        new_route1 = routes[t1].copy()
                        new_route2 = routes[t2].copy()
                        new_route1[i], new_route2[j] = new_route2[j], new_route1[i]
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        other_max = max(route_dists[t] for t in range(truck_count) if t != t1 and t != t2)
                        new_max = max(new_dist1, new_dist2, other_max)
                        if new_max < max_dist - 1e-9:
                            routes[t1] = new_route1
                            routes[t2] = new_route2
                            route_dists[t1] = new_dist1
                            route_dists[t2] = new_dist2
                            max_dist = new_max
                            full_routes = [[0] + r + [0] for r in routes]
                            report_best_vrp(full_routes)
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

    final_routes = [[0] + r + [0] for r in routes]
    return final_routes