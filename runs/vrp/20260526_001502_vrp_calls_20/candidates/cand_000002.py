import numpy as np
from itertools import permutations

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    unvisited = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    route_lengths = [0.0] * truck_count

    # Helper: compute cost of inserting customer into route at given position
    def insertion_cost(route, customer, pos):
        # pos is index where to insert, between 0 and len(route)-1 (since route has 0 at both ends)
        cost = (distance_matrix[route[pos-1]][customer] +
                distance_matrix[customer][route[pos]] -
                distance_matrix[route[pos-1]][route[pos]])
        return cost

    # Cheapest insertion of all customers
    while unvisited:
        best_cost = float('inf')
        best_route = None
        best_pos = None
        best_cust = None
        for cust in unvisited:
            for r in range(truck_count):
                route = routes[r]
                # positions where we can insert (1..len(route)-1)
                for pos in range(1, len(route)):
                    cost = insertion_cost(route, cust, pos)
                    if cost < best_cost:
                        best_cost = cost
                        best_route = r
                        best_pos = pos
                        best_cust = cust
        # Insert best customer
        routes[best_route].insert(best_pos, best_cust)
        route_lengths[best_route] += distance_matrix[routes[best_route][best_pos-1]][
            best_cust] + distance_matrix[best_cust][routes[best_route][best_pos+1]] - \
            distance_matrix[routes[best_route][best_pos-1]][routes[best_route][best_pos+1]]
        unvisited.remove(best_cust)

    # Compute route lengths properly
    for r in range(truck_count):
        route = routes[r]
        dist = 0.0
        for i in range(len(route)-1):
            dist += distance_matrix[route[i]][route[i+1]]
        route_lengths[r] = dist

    best_routes = [r[:] for r in routes]
    best_max = max(route_lengths)

    # Improvement: relocate from longest route to others + 2-opt
    max_iter = n * truck_count
    for _ in range(max_iter):
        # Find longest route
        max_len = max(route_lengths)
        if max_len == 0:
            break
        longest_routes = [i for i, l in enumerate(route_lengths) if l == max_len]
        improved = False
        for r in longest_routes:
            route = routes[r]
            # Try to relocate each customer (except depot)
            for idx in range(1, len(route)-1):
                cust = route[idx]
                # Remove customer
                removed_cost = (distance_matrix[route[idx-1]][cust] +
                                distance_matrix[cust][route[idx+1]] -
                                distance_matrix[route[idx-1]][route[idx+1]])
                new_route = route[:idx] + route[idx+1:]
                new_len_original = route_lengths[r] - removed_cost
                # Try inserting into other routes
                for r2 in range(truck_count):
                    if r2 == r:
                        continue
                    route2 = routes[r2]
                    for pos in range(1, len(route2)):
                        cost_ins = insertion_cost(route2, cust, pos)
                        new_len_r2 = route_lengths[r2] + cost_ins
                        new_max = max(new_len_original, new_len_r2)
                        if new_max < best_max:
                            # Apply move
                            routes[r] = new_route
                            route_lengths[r] = new_len_original
                            routes[r2].insert(pos, cust)
                            route_lengths[r2] = new_len_r2
                            # Recompute best
                            best_max = max(route_lengths)
                            best_routes = [r[:] for r in routes]
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
        # If no relocation improved, try 2-opt on each route
        for r in range(truck_count):
            route = routes[r]
            if len(route) <= 3:
                continue
            improved_2opt = True
            while improved_2opt:
                improved_2opt = False
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        old_dist = distance_matrix[route[i-1]][route[i]] + distance_matrix[route[j]][route[j+1]]
                        new_dist = distance_matrix[route[i-1]][route[j]] + distance_matrix[route[i]][route[j+1]]
                        if new_dist < old_dist - 1e-9:
                            # Reverse segment from i to j
                            route[i:j+1] = reversed(route[i:j+1])
                            # Update route_lengths
                            new_len = 0.0
                            for k in range(len(route)-1):
                                new_len += distance_matrix[route[k]][route[k+1]]
                            route_lengths[r] = new_len
                            improved_2opt = True
                            if new_len < best_max:
                                best_max = max(route_lengths)
                                best_routes = [r[:] for r in routes]
                                break
                    if improved_2opt:
                        break
        # Update best_routes after 2-opt
        current_max = max(route_lengths)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            improved = True
        if not improved:
            break

    # Ensure empty trucks
    for r in range(truck_count):
        if len(best_routes[r]) == 0:
            best_routes[r] = [0, 0]
    return best_routes