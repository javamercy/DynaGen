import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    n_cust = n - 1
    if truck_count >= n_cust:
        routes = [[0, 0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i-1] = [0, i, 0]
        return routes

    # Farthest-first seeding
    centers = [np.argmax(distance_matrix[0, 1:]) + 1]
    for _ in range(1, truck_count):
        dist_to_centers = np.min([[distance_matrix[c, i] for i in range(1, n)] for c in centers], axis=0)
        new_center = np.argmax(dist_to_centers) + 1
        centers.append(new_center)
    centers = np.array(centers)

    # Assign each customer to nearest center
    clusters = [[] for _ in range(truck_count)]
    for cust in range(1, n):
        dists = [distance_matrix[center, cust] for center in centers]
        cluster_idx = np.argmin(dists)
        clusters[cluster_idx].append(cust)

    # Nearest neighbor + 2-opt
    def nearest_neighbor_route(nodes):
        route = [0]
        remaining = set(nodes)
        current = 0
        while remaining:
            next_node = min(remaining, key=lambda x: distance_matrix[current, x])
            route.append(next_node)
            remaining.remove(next_node)
            current = next_node
        route.append(0)
        return route

    def route_distance(route):
        total = 0
        for i in range(len(route)-1):
            total += distance_matrix[route[i], route[i+1]]
        return total

    def two_opt(route):
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_distance(new_route) < route_distance(best):
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best

    # Build initial routes
    routes = []
    for cluster in clusters:
        if not cluster:
            routes.append([0, 0])
        else:
            route = nearest_neighbor_route(cluster)
            route = two_opt(route)
            routes.append(route)

    best_routes = [r[:] for r in routes]
    report_best_vrp(best_routes)

    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)

    def relocate_move(routes, from_route, to_route, cust_pos, insert_pos):
        new_routes = [r[:] for r in routes]
        cust = new_routes[from_route].pop(cust_pos)
        # If the route becomes only depot, set to [0,0] (already handled by pop)
        new_routes[to_route].insert(insert_pos, cust)
        return new_routes

    def exchange_move(routes, r1, p1, r2, p2):
        new_routes = [r[:] for r in routes]
        cust1 = new_routes[r1][p1]
        cust2 = new_routes[r2][p2]
        new_routes[r1][p1] = cust2
        new_routes[r2][p2] = cust1
        return new_routes

    max_iter = n_cust * 5
    for _ in range(max_iter):
        moved = False
        # Relocate moves
        for from_route in range(truck_count):
            for to_route in range(truck_count):
                if from_route == to_route or len(routes[from_route]) <= 2:
                    continue
                for cust_pos in range(1, len(routes[from_route])-1):
                    for insert_pos in range(1, len(routes[to_route])):
                        new_routes = relocate_move(routes, from_route, to_route, cust_pos, insert_pos)
                        new_max = max_route_distance(new_routes)
                        if new_max <= max_route_distance(best_routes):
                            routes = new_routes
                            routes[from_route] = two_opt(routes[from_route])
                            routes[to_route] = two_opt(routes[to_route])
                            if new_max < max_route_distance(best_routes):
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    break
            if moved:
                break
        if moved:
            continue
        # Exchange moves
        for r1, r2 in combinations(range(truck_count), 2):
            if len(routes[r1]) <= 2 or len(routes[r2]) <= 2:
                continue
            for p1 in range(1, len(routes[r1])-1):
                for p2 in range(1, len(routes[r2])-1):
                    new_routes = exchange_move(routes, r1, p1, r2, p2)
                    new_max = max_route_distance(new_routes)
                    if new_max <= max_route_distance(best_routes):
                        routes = new_routes
                        routes[r1] = two_opt(routes[r1])
                        routes[r2] = two_opt(routes[r2])
                        if new_max < max_route_distance(best_routes):
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break
        if not moved:
            break
        # Optional: within-route 2-opt for all routes
        for idx in range(truck_count):
            routes[idx] = two_opt(routes[idx])
        if max_route_distance(routes) < max_route_distance(best_routes):
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)

    return best_routes