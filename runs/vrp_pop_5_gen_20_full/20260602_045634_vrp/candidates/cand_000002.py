import numpy as np

def solve_vrp(distance_matrix, truck_count):
    def route_distance(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    distances = [0.0] * truck_count
    customers = list(range(1, n))

    # Greedy insertion
    for cust in customers:
        best_increase = float('inf')
        best_route = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            # Try positions from 1 to len(route)-1 (before the final 0)
            for pos in range(1, len(route)):
                new_dist = distances[r] \
                    + distance_matrix[route[pos-1], cust] \
                    + distance_matrix[cust, route[pos]] \
                    - distance_matrix[route[pos-1], route[pos]]
                # New max distance
                new_max = max(new_dist, max(distances[:r] + distances[r+1:]))
                increase = new_max - max(distances)
                # if increase becomes negative? actually it can't be negative but we compute anyway
                if (increase < best_increase) or (increase == best_increase and r < best_route):
                    best_increase = increase
                    best_route = r
                    best_pos = pos
        # Insert
        route = routes[best_route]
        route.insert(best_pos, cust)
        distances[best_route] = route_distance(route)

    # Initial best
    best_routes = [list(r) for r in routes]
    best_max = max(distances)
    report_best_vrp(best_routes)

    # Helper to evaluate max if we modify a route
    def eval_change(r, new_route):
        new_dist = route_distance(new_route)
        return max(new_dist, max(distances[:r] + distances[r+1:]))

    # Improvement: 2-opt on each route
    for route_idx in range(truck_count):
        route = routes[route_idx]
        improved = True
        max_iter = len(route) * 2  # bounded
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    # Reverse segment from i to j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_max = eval_change(route_idx, new_route)
                    if new_max < max(distances):
                        routes[route_idx] = new_route
                        distances[route_idx] = route_distance(new_route)
                        improved = True
                        if new_max < best_max:
                            best_routes = [list(r) for r in routes]
                            best_max = new_max
                            report_best_vrp(best_routes)
                        break  # restart from beginning
                if improved:
                    break

    # Improvement: relocate (move a customer to another route)
    max_iter = n * truck_count  # bounded
    iteration = 0
    improved = True
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for src in range(truck_count):
            route_src = routes[src]
            for pos_src in range(1, len(route_src)-1):
                cust = route_src[pos_src]
                # Remove cust from src
                temp_src = route_src[:pos_src] + route_src[pos_src+1:]
                new_dist_src = route_distance(temp_src)
                for dst in range(truck_count):
                    if dst == src:
                        continue
                    route_dst = routes[dst]
                    for pos_dst in range(1, len(route_dst)):
                        new_route_dst = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                        new_max = max(new_dist_src, route_distance(new_route_dst), 
                                      max([distances[i] for i in range(truck_count) if i != src and i != dst]))
                        if new_max < max(distances):
                            # Apply change
                            routes[src] = temp_src
                            distances[src] = new_dist_src
                            routes[dst] = new_route_dst
                            distances[dst] = route_distance(new_route_dst)
                            improved = True
                            if new_max < best_max:
                                best_routes = [list(r) for r in routes]
                                best_max = new_max
                                report_best_vrp(best_routes)
                            break  # go to next iteration
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_routes