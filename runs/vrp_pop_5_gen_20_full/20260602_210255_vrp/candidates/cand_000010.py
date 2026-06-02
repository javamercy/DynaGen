import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Trivial case: enough trucks for each customer
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        routes += [[0, 0]] * (truck_count - (n - 1))
        return routes
    
    # Step 1: Initialize seeds using farthest-point clustering
    seed_customers = []
    # first seed: farthest from depot
    farest = max(customers, key=lambda c: distance_matrix[0][c])
    seed_customers.append(farest)
    
    # remaining seeds: farthest from any selected seed
    while len(seed_customers) < truck_count:
        best_cust = None
        best_min_dist = -1
        for c in customers:
            if c in seed_customers:
                continue
            min_dist = min(distance_matrix[c][s] for s in seed_customers)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_cust = c
        seed_customers.append(best_cust)
    
    # Create initial routes with seeds
    routes = [[0, s, 0] for s in seed_customers]
    route_dist = [distance_matrix[0][s] + distance_matrix[s][0] for s in seed_customers]
    assigned = set(seed_customers)
    
    # Step 2: Insert remaining customers
    remaining = [c for c in customers if c not in assigned]
    # sort by distance from depot descending (farthest first)
    remaining.sort(key=lambda c: distance_matrix[0][c], reverse=True)
    
    for cust in remaining:
        best_new_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        # Evaluate each route to minimize new overall max
        for idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                next = route[pos]
                new_route_dist = route_dist[idx] - distance_matrix[prev][next] + distance_matrix[prev][cust] + distance_matrix[cust][next]
                new_overall_max = max(route_dist[:idx] + [new_route_dist] + route_dist[idx+1:])
                if new_overall_max < best_new_max:
                    best_new_max = new_overall_max
                    best_route_idx = idx
                    best_pos = pos
        # Insert at best position
        route = routes[best_route_idx]
        pos = best_pos
        prev = route[pos-1]
        next = route[pos]
        route_dist[best_route_idx] += -distance_matrix[prev][next] + distance_matrix[prev][cust] + distance_matrix[cust][next]
        route.insert(pos, cust)
        assigned.add(cust)
    
    # Step 3: Intra-route 2-opt improvement
    def route_distance(route):
        dist = 0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i]][route[i+1]]
        return dist
    
    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in routes)
    
    # Number of iterations based on instance size
    max_iter = max(50, n * 2)
    for _ in range(max_iter):
        improved = False
        for idx, route in enumerate(routes):
            if len(route) <= 4:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j == i+1:
                        continue
                    # 2-opt: reverse segment i to j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route)
                    if new_dist < route_distance(route):
                        routes[idx] = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            current_max = max(route_distance(r) for r in routes)
            if current_max < best_max:
                best_routes = [r[:] for r in routes]
                best_max = current_max
        else:
            # No improvement after full pass
            break
    
    return best_routes[:truck_count]