import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    
    # Seed randomness from instance for reproducibility
    seed = int(np.sum(distance_matrix) * 1e6) % (2**31)
    random.seed(seed)
    
    customers = list(range(1, n))
    num_customers = n - 1
    k = min(truck_count, num_customers)
    
    # Random medoid selection with probability proportional to distance to nearest medoid
    medoids = []
    first = random.choice(customers)
    medoids.append(first)
    while len(medoids) < k:
        dists = []
        for c in customers:
            min_dist = min(distance_matrix[c][m] for m in medoids)
            dists.append(min_dist)
        # weight by squared distance to favor far points
        total = sum(d for d in dists)
        if total == 0:
            weights = [1/len(dists)] * len(dists)
        else:
            weights = [d / total for d in dists]
        next_c = random.choices(customers, weights=weights, k=1)[0]
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
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    
    def tsp_random_start(cluster):
        if not cluster:
            return [0, 0]
        # Random start customer from cluster
        start = random.choice(cluster)
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
    routes = [tsp_random_start(cluster) for cluster in clusters]
    best_max_dist = max(route_dist(r) for r in routes)
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)
    
    # Simulated annealing improvement
    current_max = best_max_dist
    # Initial temperature: 10% of average route distance
    avg_dist = sum(route_dist(r) for r in routes) / truck_count
    T = max(avg_dist * 0.1, 0.1)
    cooling_rate = 0.99
    max_iterations = num_customers * truck_count * 10
    for iteration in range(max_iterations):
        # Pick a random customer from a non-empty route (excluding depot)
        non_empty = [i for i, r in enumerate(routes) if len(r) > 2]
        if not non_empty:
            break
        src_idx = random.choice(non_empty)
        src_route = routes[src_idx]
        # Random customer position (indices 1..len-2)
        pos_idx = random.randint(1, len(src_route)-2)
        cust = src_route[pos_idx]
        # Choose a different route as destination (could be empty)
        dst_idx = random.randint(0, truck_count-1)
        while dst_idx == src_idx:
            dst_idx = random.randint(0, truck_count-1)
        dst_route = routes[dst_idx]
        # Remove customer from source
        new_src = src_route[:pos_idx] + src_route[pos_idx+1:]
        # Insert into destination at best position (by route distance)
        best_pos = None
        best_dst_dist = None
        for p in range(1, len(dst_route)):
            new_dst = dst_route[:p] + [cust] + dst_route[p:]
            d = route_dist(new_dst)
            if best_dst_dist is None or d < best_dst_dist:
                best_dst_dist = d
                best_pos = p
        if best_pos is None:
            continue
        new_dst = dst_route[:best_pos] + [cust] + dst_route[best_pos:]
        # Compute new max distance
        dist_src = route_dist(new_src)
        dist_dst = route_dist(new_dst)
        max_rest = 0.0
        for j, r in enumerate(routes):
            if j not in (src_idx, dst_idx):
                d = route_dist(r)
                if d > max_rest:
                    max_rest = d
        new_max = max(dist_src, dist_dst, max_rest)
        # Accept if improves or with probability
        delta = new_max - current_max
        if delta < 0 or random.random() < np.exp(-delta / T):
            routes[src_idx] = new_src
            routes[dst_idx] = new_dst
            current_max = new_max
            if current_max < best_max_dist:
                best_max_dist = current_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
        # Cool down
        T *= cooling_rate
    
    # Ensure exactly truck_count routes
    while len(best_routes) < truck_count:
        best_routes.append([0, 0])
    return best_routes