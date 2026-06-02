import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [0.0 for _ in range(truck_count)]
    
    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    for r in range(truck_count):
        route_distances[r] = compute_route_distance(routes[r])
    
    unassigned = set(range(1, n))
    
    # Construction: regret-2 insertion
    while unassigned:
        best_customer = None
        best_regret = -1.0
        best_pos = None
        for c in unassigned:
            data = []
            for r_idx, route in enumerate(routes):
                curr_dist = route_distances[r_idx]
                for i in range(1, len(route)):
                    new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], c] + distance_matrix[c, route[i]]
                    other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                    cand_max = max(new_dist, other_max)
                    data.append((cand_max, r_idx, i))
            data.sort(key=lambda x: (x[0], x[1], x[2]))
            if len(data) >= 2:
                regret = data[1][0] - data[0][0]
            else:
                regret = 0.0
            if regret > best_regret or (regret == best_regret and c < best_customer):
                best_regret = regret
                best_customer = c
                best_pos = (data[0][1], data[0][2])
        r_idx, i = best_pos
        route = routes[r_idx]
        route.insert(i, best_customer)
        route_distances[r_idx] = compute_route_distance(route)
        unassigned.remove(best_customer)
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_distances)
    
    def report_best_vrp(routes):
        nonlocal best_routes, best_max
        current_max = max(route_distances)
        if current_max < best_max:
            best_max = current_max
            best_routes = [list(r) for r in routes]
    
    report_best_vrp(routes)
    
    # VND loops
    def vnd(routes, route_distances):
        improved = True
        iters = 0
        max_iters = n
        while improved and iters < max_iters:
            iters += 1
            improved = False
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                best_route = route[:]
                best_dist = route_distances[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_route = new_route[:]
                if best_dist < route_distances[r_idx]:
                    routes[r_idx] = best_route
                    route_distances[r_idx] = best_dist
                    improved = True
                    report_best_vrp(routes)
            # Inter-route swap
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max(route_distances[:r1] + route_distances[r1+1:r2] + route_distances[r2+1:], default=0.0)
                            new_max = max(new_dist1, new_dist2, other_max)
                            if new_max < best_max:
                                routes[r1] = new1
                                routes[r2] = new2
                                route_distances[r1] = new_dist1
                                route_distances[r2] = new_dist2
                                improved = True
                                report_best_vrp(routes)
        return routes, route_distances
    
    routes, route_distances = vnd(routes, route_distances)
    report_best_vrp(routes)
    
    # Perturbation and restart (up to 3 times)
    for _ in range(3):
        # Check if best solution improved from last perturbation
        prev_best = best_max
        # Remove a deterministic subset of customers (first 10% by index)
        k = max(1, n // 10)
        to_remove = set(range(1, k+1))
        removed_routes = []
        for r_idx in range(truck_count):
            route = routes[r_idx]
            new_route = [0]
            for node in route[1:-1]:
                if node in to_remove:
                    removed_routes.append((r_idx, new_route[-1], node))
                else:
                    new_route.append(node)
            new_route.append(0)
            routes[r_idx] = new_route
            route_distances[r_idx] = compute_route_distance(new_route)
        # Reinsert removed customers using regret-2
        unassigned = set([node for _, _, node in removed_routes])
        # Sort removed by original position (optional, but keep deterministic)
        removed_sorted = sorted(removed_routes, key=lambda x: (x[0], x[1], x[2]))
        while unassigned:
            best_customer = None
            best_regret = -1.0
            best_pos = None
            for c in unassigned:
                data = []
                for r_idx, route in enumerate(routes):
                    curr_dist = route_distances[r_idx]
                    for i in range(1, len(route)):
                        new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], c] + distance_matrix[c, route[i]]
                        other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                        cand_max = max(new_dist, other_max)
                        data.append((cand_max, r_idx, i))
                data.sort(key=lambda x: (x[0], x[1], x[2]))
                if len(data) >= 2:
                    regret = data[1][0] - data[0][0]
                else:
                    regret = 0.0
                if regret > best_regret or (regret == best_regret and c < best_customer):
                    best_regret = regret
                    best_customer = c
                    best_pos = (data[0][1], data[0][2])
            r_idx, i = best_pos
            route = routes[r_idx]
            route.insert(i, best_customer)
            route_distances[r_idx] = compute_route_distance(route)
            unassigned.remove(best_customer)
        # Apply VND again
        routes, route_distances = vnd(routes, route_distances)
        report_best_vrp(routes)
        # If no improvement, break early (optional)
        if best_max == prev_best:
            break
    
    return best_routes