import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    dist = distance_matrix

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d

    # 1. Seed selection (farthest-first)
    seeds = []
    # first seed: farthest from depot, tie-break smallest index
    candidates = list(range(1, n))
    first = max(candidates, key=lambda i: (dist[0, i], -i))
    seeds.append(first)
    remaining = [c for c in candidates if c != first]
    for _ in range(truck_count-1):
        if not remaining:
            break
        def min_dist(cust):
            return min(dist[cust][s] for s in seeds)
        best = max(remaining, key=lambda i: (min_dist(i), -i))
        seeds.append(best)
        remaining.remove(best)

    # 2. Initial assignment: assign each customer to nearest seed
    clusters = {s: [s] for s in seeds}
    for cust in remaining:
        nearest = min(seeds, key=lambda s: (dist[cust][s], s))
        clusters[nearest].append(cust)

    # 3. Build initial routes via nearest neighbor TSP
    routes = [[] for _ in range(truck_count)]
    route_dists = [0.0]*truck_count
    seed_list = list(seeds)
    for idx, seed in enumerate(seed_list):
        cluster = clusters[seed]
        tour = [0]
        unvisited = set(cluster)
        while unvisited:
            current = tour[-1]
            next_cust = min(unvisited, key=lambda i: (dist[current][i], i))
            tour.append(next_cust)
            unvisited.remove(next_cust)
        tour.append(0)
        routes[idx] = tour
        route_dists[idx] = route_distance(tour)
    for idx in range(len(seed_list), truck_count):
        routes[idx] = [0,0]
        route_dists[idx] = 0.0

    full_routes = [list(r) for r in routes]
    report_best_vrp(full_routes)
    current_max = max(route_dists)

    # 4. Improvement: balancing by moving customers from max route
    max_iter = 10 * n * truck_count
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        max_idx = max(range(truck_count), key=lambda i: (route_dists[i], -i))
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            continue
        # Relocate a customer from max route to another route
        for cust in max_route[1:-1]:
            new_max_route = [0] + [c for c in max_route[1:-1] if c != cust] + [0]
            new_max_dist = route_distance(new_max_route)
            for t in range(truck_count):
                if t == max_idx:
                    continue
                other_route = routes[t]
                best_pos = None
                best_new_other_dist = None
                best_new_max = float('inf')
                for pos in range(1, len(other_route)):
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    new_other_dist = route_distance(new_other)
                    other_max = max([route_dists[i] for i in range(truck_count) if i not in (max_idx, t)], default=0.0)
                    potential_max = max(new_max_dist, new_other_dist, other_max)
                    if potential_max < best_new_max or (potential_max == best_new_max and (best_pos is None or pos < best_pos)):
                        best_new_max = potential_max
                        best_pos = pos
                        best_new_other_dist = new_other_dist
                if best_new_max < current_max:
                    routes[max_idx] = new_max_route
                    route_dists[max_idx] = new_max_dist
                    routes[t] = routes[t][:best_pos] + [cust] + routes[t][best_pos:]
                    route_dists[t] = best_new_other_dist
                    current_max = best_new_max
                    improved = True
                    full_routes = [list(r) for r in routes]
                    report_best_vrp(full_routes)
                    break
            if improved:
                break
        if improved:
            continue
        # Swap between max route and another route
        for i in range(1, len(max_route)-1):
            cust1 = max_route[i]
            for t in range(truck_count):
                if t == max_idx or len(routes[t]) <= 2:
                    continue
                for j in range(1, len(routes[t])-1):
                    cust2 = routes[t][j]
                    new_route1 = max_route[:i] + [cust2] + max_route[i+1:]
                    new_route2 = routes[t][:j] + [cust1] + routes[t][j+1:]
                    new_dist1 = route_distance(new_route1)
                    new_dist2 = route_distance(new_route2)
                    other_max = max([route_dists[i] for i in range(truck_count) if i not in (max_idx, t)], default=0.0)
                    new_max = max(new_dist1, new_dist2, other_max)
                    if new_max < current_max:
                        routes[max_idx] = new_route1
                        routes[t] = new_route2
                        route_dists[max_idx] = new_dist1
                        route_dists[t] = new_dist2
                        current_max = new_max
                        improved = True
                        full_routes = [list(r) for r in routes]
                        report_best_vrp(full_routes)
                        break
                if improved:
                    break
            if improved:
                break

    final_routes = [list(r) for r in routes]
    return final_routes