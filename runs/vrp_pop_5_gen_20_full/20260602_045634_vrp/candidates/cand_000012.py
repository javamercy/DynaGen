import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    customers = list(range(1, n))
    num_customers = n - 1
    k = min(truck_count, num_customers)
    
    # Farthest-first medoid selection (deterministic tie-breaking by index)
    medoids = []
    first_medoid = max(customers, key=lambda c: (distance_matrix[0][c], -c))
    medoids.append(first_medoid)
    while len(medoids) < k:
        dist_to_medoids = []
        for c in customers:
            min_dist = min(distance_matrix[c][m] for m in medoids)
            dist_to_medoids.append((min_dist, -c))
        next_c = max(customers, key=lambda c: (min(distance_matrix[c][m] for m in medoids), -c))
        medoids.append(next_c)
    
    # Assign customers to nearest medoid (tie-break by medoid index)
    assignment = {m: [] for m in medoids}
    for c in customers:
        nearest = min(medoids, key=lambda m: (distance_matrix[c][m], m))
        assignment[nearest].append(c)
    clusters = list(assignment.values())
    while len(clusters) < truck_count:
        clusters.append([])
    
    def route_dist(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    def tsp(cluster):
        if not cluster:
            return [0, 0]
        # Start from the customer farthest from depot (deterministic)
        start = max(cluster, key=lambda c: (distance_matrix[0][c], -c))
        route = [0, start]
        unvisited = set(cluster)
        unvisited.remove(start)
        current = start
        while unvisited:
            next_c = min(unvisited, key=lambda c: (distance_matrix[current][c], c))
            route.append(next_c)
            unvisited.remove(next_c)
            current = next_c
        route.append(0)
        # 2-opt improvement (bounded)
        improved = True
        max_iter = len(cluster) * 5
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_dist(new_route) < route_dist(route):
                        route = new_route
                        improved = True
        return route
    
    # Build initial routes
    routes = [tsp(cluster) for cluster in clusters]
    best_max_dist = max(route_dist(r) for r in routes)
    report_best_vrp(routes)
    
    # Improvement: relocate from longest to shortest (including empty routes)
    max_global_iter = num_customers * truck_count * 2
    while max_global_iter > 0:
        max_global_iter -= 1
        # Find longest route
        max_dist = max(route_dist(r) for r in routes)
        longest_idx = None
        for idx, r in enumerate(routes):
            if route_dist(r) == max_dist:
                longest_idx = idx
                break
        if longest_idx is None:
            break
        longest_route = routes[longest_idx]
        if len(longest_route) <= 2:
            break
        # Find best move: relocate a customer from longest to another route
        best_move = None
        best_new_max = max_dist
        for cust_idx in range(1, len(longest_route)-1):
            cust = longest_route[cust_idx]
            new_long = longest_route[:cust_idx] + longest_route[cust_idx+1:]
            dist_long = route_dist(new_long)
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_idx:
                    continue
                # Try insert at best position in other route
                best_other_dist = None
                best_pos = None
                for pos in range(1, len(other_route)):
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    dist_other = route_dist(new_other)
                    if (best_other_dist is None) or (dist_other < best_other_dist):
                        best_other_dist = dist_other
                        best_pos = pos
                if best_pos is None:
                    continue
                # Compute new max distance
                other_dist = best_other_dist
                max_rest = 0.0
                for j, r in enumerate(routes):
                    if j not in (longest_idx, other_idx):
                        d = route_dist(r)
                        if d > max_rest:
                            max_rest = d
                new_max_candidate = max(dist_long, other_dist, max_rest)
                if new_max_candidate < best_new_max:
                    best_new_max = new_max_candidate
                    best_move = (longest_idx, other_idx, cust_idx, best_pos, cust)
        if best_move is not None:
            li, oi, idx, pos, cust = best_move
            new_long = routes[li][:idx] + routes[li][idx+1:]
            new_other = routes[oi][:pos] + [cust] + routes[oi][pos:]
            routes[li] = new_long
            routes[oi] = new_other
            new_max = max(route_dist(r) for r in routes)
            if new_max < best_max_dist:
                best_max_dist = new_max
                report_best_vrp(routes)
        else:
            break
    # Ensure exactly truck_count routes
    while len(routes) < truck_count:
        routes.append([0, 0])
    return routes