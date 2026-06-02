import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    random.seed(0)
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    customers = list(range(1, n))
    num_customers = n - 1
    k = min(truck_count, num_customers)

    # Farthest-first initialization of medoids
    medoids = []
    first_medoid = max(customers, key=lambda c: distance_matrix[0][c])
    medoids.append(first_medoid)
    while len(medoids) < k:
        dist_to_medoids = [min(distance_matrix[c][m] for m in medoids) for c in customers]
        next_medoid = max(customers, key=lambda c: dist_to_medoids[customers.index(c)])
        medoids.append(next_medoid)

    # K-medoids clustering
    n_iter = 10
    for _ in range(n_iter):
        assignment = {m: [] for m in medoids}
        for c in customers:
            nearest = min(medoids, key=lambda m: distance_matrix[c][m])
            assignment[nearest].append(c)
        new_medoids = []
        for m in medoids:
            cluster = assignment[m]
            if cluster:
                best = min(cluster, key=lambda p: sum(distance_matrix[p][q] for q in cluster))
                new_medoids.append(best)
            else:
                new_medoids.append(m)
        if set(new_medoids) == set(medoids):
            break
        medoids = new_medoids

    final_assignment = {m: [] for m in medoids}
    for c in customers:
        nearest = min(medoids, key=lambda m: distance_matrix[c][m])
        final_assignment[nearest].append(c)
    clusters = list(final_assignment.values())
    while len(clusters) < truck_count:
        clusters.append([])

    # Helper to compute route distance
    def route_dist(route):
        if len(route) <= 1:
            return 0
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    # TSP: nearest neighbor + 2-opt
    def tsp(cluster):
        if not cluster:
            return [0, 0]
        start_points = cluster[:]
        if len(cluster) > 10:
            start_points = [max(cluster, key=lambda c: distance_matrix[0][c])]
        best_route = None
        best_dist = float('inf')
        for start in start_points:
            route = [0, start]
            unvisited = set(cluster)
            unvisited.remove(start)
            current = start
            while unvisited:
                next_c = min(unvisited, key=lambda c: distance_matrix[current][c])
                route.append(next_c)
                unvisited.remove(next_c)
                current = next_c
            route.append(0)
            # 2-opt
            improved = True
            max_iter_2opt = 100
            while improved and max_iter_2opt > 0:
                improved = False
                max_iter_2opt -= 1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_dist(new_route) < route_dist(route):
                            route = new_route
                            improved = True
            if route_dist(route) < best_dist:
                best_dist = route_dist(route)
                best_route = route
        return best_route

    # Build initial routes
    routes = [tsp(cluster) for cluster in clusters]
    best_max_dist = max(route_dist(r) for r in routes)
    report_best_vrp(routes)

    # Improvement: relocate and swap
    max_improve_iters = min(1000, n * truck_count * 5)
    for _ in range(max_improve_iters):
        max_dist = max(route_dist(r) for r in routes)
        # Find longest and shortest routes
        longest_idx = max(range(len(routes)), key=lambda i: route_dist(routes[i]))
        shortest_idx = min(range(len(routes)), key=lambda i: route_dist(routes[i]))
        if longest_idx == shortest_idx:
            break
        longest_route = routes[longest_idx]
        shortest_route = routes[shortest_idx]
        best_move = None
        best_new_max = max_dist

        # Evaluate relocate moves: move customer from longest to shortest
        for idx in range(1, len(longest_route)-1):
            cust = longest_route[idx]
            # Remove cust from longest
            new_long = longest_route[:idx] + longest_route[idx+1:]
            if new_long[0] != 0 or new_long[-1] != 0:
                continue
            dist_long = route_dist(new_long)
            # Insert into shortest at best position
            best_insert_dist = None
            best_pos = None
            for pos in range(1, len(shortest_route)):
                new_short = shortest_route[:pos] + [cust] + shortest_route[pos:]
                dist_short = route_dist(new_short)
                if best_insert_dist is None or dist_short < best_insert_dist:
                    best_insert_dist = dist_short
                    best_pos = pos
            # Compute new max
            other_max = max(route_dist(r) for i, r in enumerate(routes) if i not in [longest_idx, shortest_idx])
            new_max = max(dist_long, best_insert_dist, other_max)
            if new_max < best_new_max:
                best_new_max = new_max
                best_move = ('relocate', longest_idx, shortest_idx, idx, best_pos, cust)

        # Evaluate swap moves: exchange a customer from longest with a customer from shortest
        for i in range(1, len(longest_route)-1):
            cust1 = longest_route[i]
            for j in range(1, len(shortest_route)-1):
                cust2 = shortest_route[j]
                # Remove both
                new_long = longest_route[:i] + longest_route[i+1:]
                new_short = shortest_route[:j] + shortest_route[j+1:]
                if new_long[0] != 0 or new_long[-1] != 0 or new_short[0] != 0 or new_short[-1] != 0:
                    continue
                # Insert cust2 into new_long at best position
                best_pos_long = None
                best_dist_long = None
                for pos in range(1, len(new_long)):
                    cand = new_long[:pos] + [cust2] + new_long[pos:]
                    d = route_dist(cand)
                    if best_dist_long is None or d < best_dist_long:
                        best_dist_long = d
                        best_pos_long = pos
                # Insert cust1 into new_short at best position
                best_pos_short = None
                best_dist_short = None
                for pos in range(1, len(new_short)):
                    cand = new_short[:pos] + [cust1] + new_short[pos:]
                    d = route_dist(cand)
                    if best_dist_short is None or d < best_dist_short:
                        best_dist_short = d
                        best_pos_short = pos
                # Compute new max
                other_max = max(route_dist(r) for idx, r in enumerate(routes) if idx not in [longest_idx, shortest_idx])
                new_max = max(best_dist_long, best_dist_short, other_max)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = ('swap', longest_idx, shortest_idx, i, j, cust1, cust2, best_pos_long, best_pos_short)

        if best_move is not None:
            if best_move[0] == 'relocate':
                _, li, si, idx, pos, cust = best_move
                # Apply relocate
                new_long = routes[li][:idx] + routes[li][idx+1:]
                new_short = routes[si][:pos] + [cust] + routes[si][pos:]
                # Run short 2-opt on affected routes
                def local_2opt(route, max_local=10):
                    improved = True
                    while improved and max_local > 0:
                        improved = False
                        max_local -= 1
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                cand = route[:a] + route[a:b+1][::-1] + route[b+1:]
                                if route_dist(cand) < route_dist(route):
                                    route = cand
                                    improved = True
                        return route
                new_long = local_2opt(new_long)
                new_short = local_2opt(new_short)
                routes[li] = new_long
                routes[si] = new_short
            else:  # swap
                _, li, si, i, j, cust1, cust2, pos_long, pos_short = best_move
                new_long = routes[li][:i] + routes[li][i+1:]
                new_short = routes[si][:j] + routes[si][j+1:]
                new_long = new_long[:pos_long] + [cust2] + new_long[pos_long:]
                new_short = new_short[:pos_short] + [cust1] + new_short[pos_short:]
                def local_2opt(route, max_local=10):
                    improved = True
                    while improved and max_local > 0:
                        improved = False
                        max_local -= 1
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                cand = route[:a] + route[a:b+1][::-1] + route[b+1:]
                                if route_dist(cand) < route_dist(route):
                                    route = cand
                                    improved = True
                    return route
                new_long = local_2opt(new_long)
                new_short = local_2opt(new_short)
                routes[li] = new_long
                routes[si] = new_short
            # Update best_max_dist and report
            current_max = max(route_dist(r) for r in routes)
            if current_max < best_max_dist:
                best_max_dist = current_max
                report_best_vrp(routes)
        else:
            break

    # Ensure exactly truck_count routes
    while len(routes) < truck_count:
        routes.append([0, 0])
    return routes