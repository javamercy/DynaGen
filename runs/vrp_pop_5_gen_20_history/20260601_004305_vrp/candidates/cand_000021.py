import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers]
        for _ in range(truck_count - m):
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # Sort customers by distance from depot descending, then by index
    custs = sorted(customers, key=lambda c: (-distance_matrix[0][c], c))

    # Initial empty routes
    routes = [[0, 0] for _ in range(truck_count)]

    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    def compute_max():
        maxd = 0
        for r in routes:
            d = route_dist(r)
            if d > maxd:
                maxd = d
        return maxd

    # Insert each customer greedily
    for c in custs:
        best_max = math.inf
        best_route_idx = -1
        best_pos = -1
        for ri in range(truck_count):
            route = routes[ri]
            # evaluate insertions at positions 1..len-1 (between depots)
            for pos in range(1, len(route)):
                new_route = route[:pos] + [c] + route[pos:]
                new_dist = route_dist(new_route)
                # compute new max if this insertion is chosen
                other_max = 0
                for rj in range(truck_count):
                    if rj == ri:
                        continue
                    d = route_dist(routes[rj])
                    if d > other_max:
                        other_max = d
                cur_max = max(new_dist, other_max)
                if cur_max < best_max or (cur_max == best_max and (ri < best_route_idx or (ri == best_route_idx and pos < best_pos))):
                    best_max = cur_max
                    best_route_idx = ri
                    best_pos = pos
        # Apply best insertion
        routes[best_route_idx].insert(best_pos, c)
    
    # Initial best
    best_routes = [list(r) for r in routes]
    best_max = compute_max()
    report_best_vrp(best_routes)

    current_routes = [list(r) for r in routes]
    current_max = best_max

    # RRT loop
    max_iter = 100 * n
    deviation = 0.1 * best_max  # constant threshold
    for iteration in range(max_iter):
        # Generate neighbor by shaking
        new_routes = [list(r) for r in current_routes]
        if iteration % 2 == 0:
            # Shake 1: reverse longest route (excluding depots)
            longest_idx = max(range(truck_count), key=lambda i: (route_dist(current_routes[i]), i))
            route = new_routes[longest_idx]
            if len(route) > 3:
                inner = route[1:-1]
                inner.reverse()
                new_routes[longest_idx] = [0] + inner + [0]
        else:
            # Shake 2: relocate farthest customer (by distance from depot) from longest route
            longest_idx = max(range(truck_count), key=lambda i: (route_dist(current_routes[i]), i))
            route = new_routes[longest_idx]
            if len(route) > 3:
                # Find farthest customer in route (excluding depot)
                farthest_cust = max(route[1:-1], key=lambda c: (distance_matrix[0][c], c))
                idx_c = route.index(farthest_cust)
                # Remove it
                new_routes[longest_idx] = route[:idx_c] + route[idx_c+1:]
                # Insert into best other route (best insertion)
                best_max_local = math.inf
                best_ri = -1
                best_pos = -1
                for ri in range(truck_count):
                    if ri == longest_idx:
                        continue
                    other_route = new_routes[ri]
                    for pos in range(1, len(other_route)):
                        temp = other_route[:pos] + [farthest_cust] + other_route[pos:]
                        d = route_dist(temp)
                        other_max = 0
                        for rj in range(truck_count):
                            if rj == ri:
                                continue
                            d2 = route_dist(new_routes[rj])
                            if d2 > other_max:
                                other_max = d2
                        cur_max = max(d, other_max)
                        if cur_max < best_max_local or (cur_max == best_max_local and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                            best_max_local = cur_max
                            best_ri = ri
                            best_pos = pos
                if best_ri != -1:
                    new_routes[best_ri].insert(best_pos, farthest_cust)
        # Evaluate neighbor
        new_max = max(route_dist(r) for r in new_routes)
        if new_max < best_max:
            best_max = new_max
            best_routes = [list(r) for r in new_routes]
            current_routes = [list(r) for r in new_routes]
            current_max = new_max
            report_best_vrp(best_routes)
        elif new_max <= best_max + deviation:
            current_routes = [list(r) for r in new_routes]
            current_max = new_max
        else:
            pass  # reject
    # Ensure exactly truck_count routes (already satisfied)
    # No empty trucks except for the initial case handled earlier
    return best_routes