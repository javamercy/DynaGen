import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    # Initialize routes with depot-depot
    routes = [[depot, depot] for _ in range(truck_count)]
    route_distances = [0.0] * truck_count
    
    # Helper to compute route distance
    def compute_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d
    
    # Construction: greedy min-max insertion, customers sorted by distance to depot (ascending), then by index
    sorted_customers = sorted(customers, key=lambda c: (distance_matrix[depot][c], c))
    for cust in sorted_customers:
        best_new_max = float('inf')
        best_new_dist = float('inf')
        best_route_idx = -1
        best_pos = -1
        for r_idx in range(truck_count):
            route = routes[r_idx]
            for pos in range(1, len(route)):
                # Compute new route distance after insertion
                old_dist = route_distances[r_idx]
                new_dist = old_dist - distance_matrix[route[pos-1]][route[pos]] + distance_matrix[route[pos-1]][cust] + distance_matrix[cust][route[pos]]
                # Compute new max
                current_max = 0.0
                for j in range(truck_count):
                    if j == r_idx:
                        d = new_dist
                    else:
                        d = route_distances[j]
                    if d > current_max:
                        current_max = d
                # Update best if better
                if current_max < best_new_max or (current_max == best_new_max and new_dist < best_new_dist):
                    best_new_max = current_max
                    best_new_dist = new_dist
                    best_route_idx = r_idx
                    best_pos = pos
        # Apply insertion
        routes[best_route_idx].insert(best_pos, cust)
        route_distances[best_route_idx] = best_new_dist
    
    report_best_vrp(routes)
    
    # Improvement: local search
    max_iter = n  # bounded
    for _ in range(max_iter):
        # Compute current route distances and max
        route_dists = [compute_dist(r) for r in routes]
        current_max = max(route_dists)
        longest_idx = min(i for i, d in enumerate(route_dists) if d == current_max)
        
        best_move = None
        best_new_max = current_max
        best_new_total = sum(route_dists)
        
        # 1. Relocate: move a customer from longest to another route
        route_long = routes[longest_idx]
        if len(route_long) > 2:
            for cust_pos in range(1, len(route_long)-1):
                cust = route_long[cust_pos]
                # Remove cust from longest
                new_long = route_long[:cust_pos] + route_long[cust_pos+1:]
                new_long_dist = compute_dist(new_long)
                # Try insert into other routes
                for target_idx in range(truck_count):
                    if target_idx == longest_idx:
                        continue
                    target_route = routes[target_idx]
                    for pos in range(1, len(target_route)):
                        old_target_dist = route_dists[target_idx]
                        new_target_dist = old_target_dist - distance_matrix[target_route[pos-1]][target_route[pos]] + distance_matrix[target_route[pos-1]][cust] + distance_matrix[cust][target_route[pos]]
                        # Compute new max
                        new_max = max(current_max, new_long_dist, new_target_dist)
                        # Compute new total distance
                        new_total = sum(route_dists) - route_dists[longest_idx] - route_dists[target_idx] + new_long_dist + new_target_dist
                        # Update best
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_move = ('relocate', longest_idx, cust_pos, target_idx, pos)
        
        # 2. Swap: swap a customer from longest with a customer from another route
        if len(route_long) > 2:
            for cust_pos in range(1, len(route_long)-1):
                cust1 = route_long[cust_pos]
                for other_idx in range(truck_count):
                    if other_idx == longest_idx or len(routes[other_idx]) <= 2:
                        continue
                    other_route = routes[other_idx]
                    for other_pos in range(1, len(other_route)-1):
                        cust2 = other_route[other_pos]
                        # Compute new routes
                        new_long = route_long[:]
                        new_long[cust_pos] = cust2
                        new_long_dist = compute_dist(new_long)
                        new_other = other_route[:]
                        new_other[other_pos] = cust1
                        new_other_dist = compute_dist(new_other)
                        new_max = max(current_max, new_long_dist, new_other_dist)
                        new_total = sum(route_dists) - route_dists[longest_idx] - route_dists[other_idx] + new_long_dist + new_other_dist
                        if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                            best_new_max = new_max
                            best_new_total = new_total
                            best_move = ('swap', longest_idx, cust_pos, other_idx, other_pos)
        
        # 3. 2-opt on longest route
        if len(route_long) > 3:
            for i in range(1, len(route_long)-2):
                for j in range(i+2, len(route_long)-1):
                    new_route = route_long[:i+1] + route_long[i+1:j+1][::-1] + route_long[j+1:]
                    new_dist = compute_dist(new_route)
                    new_max = max(current_max, new_dist)
                    new_total = sum(route_dists) - route_dists[longest_idx] + new_dist
                    if new_max < best_new_max or (new_max == best_new_max and new_total < best_new_total):
                        best_new_max = new_max
                        best_new_total = new_total
                        best_move = ('2opt', longest_idx, i, j)
        
        if best_move is None:
            break
        # Apply best move
        if best_move[0] == 'relocate':
            _, long_idx, cust_pos, target_idx, pos = best_move
            cust = routes[long_idx].pop(cust_pos)
            routes[target_idx].insert(pos, cust)
            route_distances[long_idx] = compute_dist(routes[long_idx])
            route_distances[target_idx] = compute_dist(routes[target_idx])
        elif best_move[0] == 'swap':
            _, long_idx, cust_pos, other_idx, other_pos = best_move
            cust1 = routes[long_idx][cust_pos]
            cust2 = routes[other_idx][other_pos]
            routes[long_idx][cust_pos] = cust2
            routes[other_idx][other_pos] = cust1
            route_distances[long_idx] = compute_dist(routes[long_idx])
            route_distances[other_idx] = compute_dist(routes[other_idx])
        elif best_move[0] == '2opt':
            _, long_idx, i, j = best_move
            routes[long_idx] = routes[long_idx][:i+1] + routes[long_idx][i+1:j+1][::-1] + routes[long_idx][j+1:]
            route_distances[long_idx] = compute_dist(routes[long_idx])
        # Report after improvement
        report_best_vrp(routes)
    
    return routes