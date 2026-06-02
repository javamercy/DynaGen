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
                nxt = route[pos]
                new_route_dist = route_dist[idx] - distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                new_overall_max = max(route_dist[:idx] + [new_route_dist] + route_dist[idx+1:])
                if new_overall_max < best_new_max:
                    best_new_max = new_overall_max
                    best_route_idx = idx
                    best_pos = pos
        # Insert at best position
        route = routes[best_route_idx]
        pos = best_pos
        prev = route[pos-1]
        nxt = route[pos]
        route_dist[best_route_idx] += -distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
        route.insert(pos, cust)
        assigned.add(cust)
    
    # Helper to compute route distance
    def route_distance(route):
        dist = 0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i]][route[i+1]]
        return dist
    
    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in routes)
    
    # Step 3: Intra-route 2-opt improvement (bounded)
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
            break
    
    # Step 4: Inter-route improvements (relocate and exchange)
    def compute_route_dists(routes):
        return [route_distance(r) for r in routes]
    
    for _ in range(max(50, n * 2)):
        improved = False
        local_routes = [r[:] for r in best_routes]
        local_dists = compute_route_dists(local_routes)
        initial_max = max(local_dists)
        best_delta = 0
        best_move = None
        
        # Try all relocations: move a customer from route i to route j
        for i in range(truck_count):
            if len(local_routes[i]) <= 3:
                continue
            for cust in local_routes[i][1:-1]:  # skip depot
                for j in range(truck_count):
                    if i == j:
                        continue
                    for pos in range(1, len(local_routes[j])):
                        # Compute new distances
                        # Remove cust from route i
                        route_i_new = [c for c in local_routes[i] if c != cust]
                        route_i_new_dist = route_distance(route_i_new)
                        # Insert cust into route j at pos
                        prev = local_routes[j][pos-1]
                        nxt = local_routes[j][pos]
                        route_j_new = local_routes[j][:pos] + [cust] + local_routes[j][pos:]
                        route_j_new_dist = route_distance(route_j_new)
                        new_dists = local_dists[:]
                        new_dists[i] = route_i_new_dist
                        new_dists[j] = route_j_new_dist
                        new_max = max(new_dists)
                        if new_max < initial_max:
                            delta = initial_max - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('relocate', i, j, cust, pos, route_i_new, route_j_new)
        
        # Try exchange: swap segments between two routes (or simple 2-opt*)
        # Here we implement a simple exchange of two customers between routes
        for i in range(truck_count):
            if len(local_routes[i]) <= 3:
                continue
            for ci in local_routes[i][1:-1]:
                for j in range(i+1, truck_count):
                    if len(local_routes[j]) <= 3:
                        continue
                    for cj in local_routes[j][1:-1]:
                        # swap ci and cj
                        route_i_new = [c if c != ci else cj for c in local_routes[i]]
                        route_j_new = [c if c != cj else ci for c in local_routes[j]]
                        # Ensure starting/ending at depot
                        route_i_new_dist = route_distance(route_i_new)
                        route_j_new_dist = route_distance(route_j_new)
                        new_dists = local_dists[:]
                        new_dists[i] = route_i_new_dist
                        new_dists[j] = route_j_new_dist
                        new_max = max(new_dists)
                        if new_max < initial_max:
                            delta = initial_max - new_max
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('exchange', i, j, ci, cj, route_i_new, route_j_new)
        
        if best_move is not None and best_delta > 0:
            if best_move[0] == 'relocate':
                _, i, j, cust, pos, route_i_new, route_j_new = best_move
                best_routes[i] = route_i_new
                best_routes[j] = route_j_new
            else:
                _, i, j, ci, cj, route_i_new, route_j_new = best_move
                best_routes[i] = route_i_new
                best_routes[j] = route_j_new
            improved = True
            new_max = max(compute_route_dists(best_routes))
            if new_max < best_max:
                best_max = new_max
        
        if not improved:
            break
    
    return best_routes[:truck_count]