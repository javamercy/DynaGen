import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    n_cust = n - 1
    if truck_count >= n_cust:
        routes = [[0, 0] for _ in range(truck_count)]
        for i in range(1, n):
            routes[i-1] = [0, i, 0]
        report_best_vrp(routes)
        return routes

    # Farthest-first seeding
    centers = [np.argmax(distance_matrix[0, 1:]) + 1]
    for _ in range(1, truck_count):
        dist_to_centers = np.min([[distance_matrix[c, i] for i in range(1, n)] for c in centers], axis=0)
        new_center = np.argmax(dist_to_centers) + 1
        centers.append(new_center)
    centers = np.array(centers)

    # Assign each customer to nearest center (deterministic tie-breaking)
    clusters = [[] for _ in range(truck_count)]
    for cust in range(1, n):
        dists = [distance_matrix[center, cust] for center in centers]
        cluster_idx = np.argmin(dists)
        clusters[cluster_idx].append(cust)

    # Nearest neighbor route construction
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

    # Build initial routes
    routes = []
    for cluster in clusters:
        if not cluster:
            routes.append([0, 0])
        else:
            route = nearest_neighbor_route(cluster)
            routes.append(route)

    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)

    def relocate_move(routes, from_route, to_route, cust_pos, insert_pos):
        new_routes = [r[:] for r in routes]
        cust = new_routes[from_route].pop(cust_pos)
        new_routes[to_route].insert(insert_pos, cust)
        return new_routes

    def exchange_move(routes, r1, p1, r2, p2):
        new_routes = [r[:] for r in routes]
        cust1 = new_routes[r1][p1]
        cust2 = new_routes[r2][p2]
        new_routes[r1][p1] = cust2
        new_routes[r2][p2] = cust1
        return new_routes

    max_iter = n_cust * truck_count
    for _ in range(max_iter):
        moved = False
        # Find longest route index
        lengths = [route_distance(r) for r in routes]
        long_idx = np.argmax(lengths)
        # Try relocate from longest to shorter
        for to_route in range(truck_count):
            if to_route == long_idx or len(routes[long_idx]) <= 2 or len(routes[to_route]) < 2:
                continue
            for cust_pos in range(1, len(routes[long_idx])-1):
                for insert_pos in range(1, len(routes[to_route])):
                    new_routes = relocate_move(routes, long_idx, to_route, cust_pos, insert_pos)
                    new_max = max_route_distance(new_routes)
                    if new_max <= best_max:
                        routes = new_routes
                        if new_max < best_max:
                            best_max = new_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break
        if moved:
            continue
        # Try exchange between longest and another
        for r2 in range(truck_count):
            if r2 == long_idx or len(routes[long_idx]) <= 2 or len(routes[r2]) <= 2:
                continue
            for p1 in range(1, len(routes[long_idx])-1):
                for p2 in range(1, len(routes[r2])-1):
                    new_routes = exchange_move(routes, long_idx, p1, r2, p2)
                    new_max = max_route_distance(new_routes)
                    if new_max <= best_max:
                        routes = new_routes
                        if new_max < best_max:
                            best_max = new_max
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

    return best_routes