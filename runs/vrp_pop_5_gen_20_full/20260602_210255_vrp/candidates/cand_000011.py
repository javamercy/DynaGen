import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    
    # Trivial case
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in customers]
        routes += [[0, 0]] * (truck_count - (n - 1))
        return routes
    
    # Step 1: Construction
    # Farthest-point seeds
    seed_customers = []
    # first seed: farthest from depot
    far = max(customers, key=lambda c: distance_matrix[0][c])
    seed_customers.append(far)
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
    
    # Initial routes
    routes = [[0, s, 0] for s in seed_customers]
    route_dist = [distance_matrix[0][s] + distance_matrix[s][0] for s in seed_customers]
    assigned = set(seed_customers)
    
    # Greedy insertion of remaining customers, farthest first
    remaining = [c for c in customers if c not in assigned]
    remaining.sort(key=lambda c: distance_matrix[0][c], reverse=True)
    
    for cust in remaining:
        best_new_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        best_route_dist = None
        for idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev = route[pos-1]
                nxt = route[pos]
                new_route_dist = route_dist[idx] - distance_matrix[prev][nxt] + distance_matrix[prev][cust] + distance_matrix[cust][nxt]
                new_max = max(route_dist[:idx] + [new_route_dist] + route_dist[idx+1:])
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_route_idx = idx
                    best_pos = pos
                    best_route_dist = new_route_dist
                elif new_max == best_new_max:
                    # tie-break: prefer smaller total distance? Here just keep first
                    pass
        # Insert
        routes[best_route_idx].insert(best_pos, cust)
        route_dist[best_route_idx] = best_route_dist
        assigned.add(cust)
    
    best_routes = [r[:] for r in routes]
    best_max = max(route_dist)
    
    # Helper to compute route distance
    def route_dist_func(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    
    # Improvement phase: limited number of passes
    max_passes = min(100, n * 2)
    for _ in range(max_passes):
        improved = False
        # Inter-route relocate
        for i in range(len(routes)):
            route_i = routes[i]
            if len(route_i) <= 3:  # need at least one customer
                continue
            for cust_pos in range(1, len(route_i)-1):
                cust = route_i[cust_pos]
                for j in range(len(routes)):
                    if i == j:
                        continue
                    route_j = routes[j]
                    for insert_pos in range(1, len(route_j)+1):
                        # Remove cust from route_i and insert into route_j at insert_pos
                        # Compute new route distances
                        # Remove
                        prev_i = route_i[cust_pos-1]
                        next_i = route_i[cust_pos+1]
                        new_dist_i = route_dist[i] - distance_matrix[prev_i][cust] - distance_matrix[cust][next_i] + distance_matrix[prev_i][next_i]
                        # Insert in j
                        if insert_pos == len(route_j):
                            prev_j = route_j[insert_pos-1]
                            new_dist_j = route_dist[j] + distance_matrix[prev_j][cust] + distance_matrix[cust][0] - distance_matrix[prev_j][0]
                        else:
                            prev_j = route_j[insert_pos-1]
                            next_j = route_j[insert_pos]
                            new_dist_j = route_dist[j] + distance_matrix[prev_j][cust] + distance_matrix[cust][next_j] - distance_matrix[prev_j][next_j]
                        # Check feasibility: new route_i length may become 2 if only one customer? That's fine (depot-depot? Actually if only depot, length becomes 2 with depot twice? But we want at least [0,0]. If route_i has only one customer, after removal it becomes [0,0]. So we need to handle that.
                        new_route_i = route_i[:cust_pos] + route_i[cust_pos+1:]
                        if len(new_route_i) == 2:  # just depot
                            new_dist_i = 0.0
                        # New max distance
                        candidate_dists = route_dist[:]
                        candidate_dists[i] = new_dist_i
                        candidate_dists[j] = new_dist_j
                        new_max = max(candidate_dists)
                        if new_max < best_max:
                            # Apply move
                            del routes[i][cust_pos]
                            if len(routes[i]) == 2:  # only depot
                                routes[i] = [0, 0]
                            routes[j].insert(insert_pos, cust)
                            route_dist[i] = new_dist_i if len(routes[i]) > 2 else 0.0
                            route_dist[j] = new_dist_j
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            improved = True
                            # Report best
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Inter-route exchange
        for i in range(len(routes)):
            route_i = routes[i]
            if len(route_i) <= 3:
                continue
            for cust_pos_i in range(1, len(route_i)-1):
                cust_i = route_i[cust_pos_i]
                for j in range(i+1, len(routes)):
                    route_j = routes[j]
                    if len(route_j) <= 3:
                        continue
                    for cust_pos_j in range(1, len(route_j)-1):
                        cust_j = route_j[cust_pos_j]
                        # Swap cust_i and cust_j
                        # Compute new distances
                        # Remove cust_i from i
                        prev_i = route_i[cust_pos_i-1]
                        next_i = route_i[cust_pos_i+1]
                        temp_dist_i = route_dist[i] - distance_matrix[prev_i][cust_i] - distance_matrix[cust_i][next_i] + distance_matrix[prev_i][next_i]
                        # Remove cust_j from j
                        prev_j = route_j[cust_pos_j-1]
                        next_j = route_j[cust_pos_j+1]
                        temp_dist_j = route_dist[j] - distance_matrix[prev_j][cust_j] - distance_matrix[cust_j][next_j] + distance_matrix[prev_j][next_j]
                        # Insert cust_j into i at cust_pos_i
                        new_dist_i = temp_dist_i - distance_matrix[prev_i][next_i] + distance_matrix[prev_i][cust_j] + distance_matrix[cust_j][next_i]
                        # Insert cust_i into j at cust_pos_j
                        new_dist_j = temp_dist_j - distance_matrix[prev_j][next_j] + distance_matrix[prev_j][cust_i] + distance_matrix[cust_i][next_j]
                        # Handle case where route becomes only depot
                        # Actually after removal, if route had only one customer, it becomes [0,0]; but then insertion would make it [0,c,0] again; but we handle by checking lengths.
                        candidate_dists = route_dist[:]
                        candidate_dists[i] = new_dist_i
                        candidate_dists[j] = new_dist_j
                        new_max = max(candidate_dists)
                        if new_max < best_max:
                            # Apply swap
                            routes[i][cust_pos_i] = cust_j
                            routes[j][cust_pos_j] = cust_i
                            route_dist[i] = new_dist_i
                            route_dist[j] = new_dist_j
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            improved = True
                            report_best_vrp(best_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Intra-route 2-opt for each route
        for idx, route in enumerate(routes):
            if len(route) <= 4:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j == i+1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_dist_func(new_route)
                    if new_dist < route_dist[idx]:
                        routes[idx] = new_route
                        route_dist[idx] = new_dist
                        new_max = max(route_dist)
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            improved = True
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_routes