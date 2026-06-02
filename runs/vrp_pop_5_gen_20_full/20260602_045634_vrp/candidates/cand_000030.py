import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    customers = list(range(1, n))
    num_customers = n - 1
    k = min(truck_count, num_customers)
    
    # --- Deterministic k-medoids clustering (farthest-first) ---
    # Start with the farthest customer from depot (break ties by smaller index)
    depot_dists = [distance_matrix[0][c] for c in customers]
    first = max(customers, key=lambda c: (depot_dists[c-1], -c))
    medoids = [first]
    remaining = [c for c in customers if c != first]
    while len(medoids) < k:
        # For each remaining customer, compute min distance to any medoid
        best_c = None
        best_dist = -1
        for c in remaining:
            d = min(distance_matrix[c][m] for m in medoids)
            if d > best_dist or (d == best_dist and c < best_c):
                best_dist = d
                best_c = c
        medoids.append(best_c)
        remaining.remove(best_c)
    
    # Assign each customer to nearest medoid (tie: smaller medoid index)
    clusters = {m: [] for m in medoids}
    for c in customers:
        nearest = min(medoids, key=lambda m: (distance_matrix[c][m], m))
        clusters[nearest].append(c)
    
    cluster_list = [clusters.get(m, []) for m in medoids]
    while len(cluster_list) < truck_count:
        cluster_list.append([])
    
    # --- Intra-route TSP via nearest neighbor + 2-opt ---
    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    def tsp(cluster):
        if not cluster:
            return [0, 0]
        # Nearest neighbor: start from farthest customer from depot (tie: smaller index)
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
        # 2-opt
        improved = True
        iter_limit = len(cluster) * 10
        while improved and iter_limit > 0:
            improved = False
            iter_limit -= 1
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
        return route
    
    routes = [tsp(cluster) for cluster in cluster_list]
    current_max = max(route_distance(r) for r in routes)
    best_max = current_max
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)
    
    # --- Inter-route improvement: relocate and swap ---
    improved = True
    max_iter = num_customers * truck_count * 2
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        dists = [route_distance(r) for r in routes]
        max_dist = max(dists)
        # Choose longest route (tie: smallest index)
        longest_idx = min(i for i, d in enumerate(dists) if d == max_dist)
        longest_route = routes[longest_idx]
        if len(longest_route) <= 2:
            break
        best_move = None
        best_new_max = max_dist
        # --- Relocate ---
        for cust_idx in range(1, len(longest_route) - 1):
            cust = longest_route[cust_idx]
            new_long = longest_route[:cust_idx] + longest_route[cust_idx+1:]
            dist_long = route_distance(new_long)
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_idx:
                    continue
                # Find best position in other route (tie: first encountered)
                best_other_dist = None
                best_pos = None
                for pos in range(1, len(other_route)):
                    new_other = other_route[:pos] + [cust] + other_route[pos:]
                    dist_other = route_distance(new_other)
                    if best_other_dist is None or dist_other < best_other_dist:
                        best_other_dist = dist_other
                        best_pos = pos
                if best_pos is None:
                    continue
                other_dist = best_other_dist
                # Compute max among rest
                max_rest = 0.0
                for j, r in enumerate(routes):
                    if j not in (longest_idx, other_idx):
                        d = route_distance(r)
                        if d > max_rest:
                            max_rest = d
                new_max = max(dist_long, other_dist, max_rest)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = ('relocate', longest_idx, other_idx, cust_idx, best_pos, cust)
        # --- Swap (only if no relocate improved) ---
        if best_move is None:
            for other_idx, other_route in enumerate(routes):
                if other_idx == longest_idx:
                    continue
                if len(other_route) <= 2:
                    continue
                for cust_idx in range(1, len(longest_route) - 1):
                    cust = longest_route[cust_idx]
                    for other_cust_idx in range(1, len(other_route) - 1):
                        other_cust = other_route[other_cust_idx]
                        # Swap customers in place
                        new_long = longest_route[:cust_idx] + [other_cust] + longest_route[cust_idx+1:]
                        new_other = other_route[:other_cust_idx] + [cust] + other_route[other_cust_idx+1:]
                        dist_long = route_distance(new_long)
                        dist_other = route_distance(new_other)
                        max_rest = 0.0
                        for j, r in enumerate(routes):
                            if j not in (longest_idx, other_idx):
                                d = route_distance(r)
                                if d > max_rest:
                                    max_rest = d
                        new_max = max(dist_long, dist_other, max_rest)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ('swap', longest_idx, other_idx, cust_idx, other_cust_idx, cust, other_cust)
        # Apply best move if found
        if best_move is not None:
            if best_move[0] == 'relocate':
                _, li, oi, ci, pos, cust = best_move
                new_long = routes[li][:ci] + routes[li][ci+1:]
                new_other = routes[oi][:pos] + [cust] + routes[oi][pos:]
                routes[li] = new_long
                routes[oi] = new_other
            else:  # swap
                _, li, oi, ci, oci, cust, other_cust = best_move
                new_long = routes[li][:ci] + [other_cust] + routes[li][ci+1:]
                new_other = routes[oi][:oci] + [cust] + routes[oi][oci+1:]
                routes[li] = new_long
                routes[oi] = new_other
            new_max = max(route_distance(r) for r in routes)
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
            improved = True
    
    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes