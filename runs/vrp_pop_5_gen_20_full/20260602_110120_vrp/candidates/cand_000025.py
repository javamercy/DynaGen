import numpy as np
import math

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
    
    # Adaptive regret depth
    k = min(max(3, int(math.log2(n))), n-1)
    
    def get_insertion_data(customer):
        data = []
        for r_idx, route in enumerate(routes):
            curr_dist = route_distances[r_idx]
            for i in range(1, len(route)):
                new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], customer] + distance_matrix[customer, route[i]]
                other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0.0)
                cand_max = max(new_dist, other_max)
                data.append((cand_max, (r_idx, i)))
        data.sort(key=lambda x: x[0])
        return data
    
    while unassigned:
        regrets = []
        for c in unassigned:
            data = get_insertion_data(c)
            m = len(data)
            if m >= k:
                regret = sum(data[i][0] - data[0][0] for i in range(1, k))
            elif m > 1:
                regret = data[1][0] - data[0][0]
            else:
                regret = 0.0
            # Tie-breaking: closest to depot (smaller distance)
            tie_breaker = distance_matrix[0, c]
            regrets.append((regret, tie_breaker, c, data[0][1]))
        # Sort by regret descending, then tie_breaker ascending
        regrets.sort(key=lambda x: (-x[0], x[1]))
        selected = regrets[0][2]
        best_pos = regrets[0][3]
        r_idx, i = best_pos
        route = routes[r_idx]
        route.insert(i, selected)
        route_distances[r_idx] = compute_route_distance(route)
        unassigned.remove(selected)
    
    current_routes = [list(r) for r in routes]
    current_max = max(route_distances)
    
    def report_best_vrp(routes):
        pass
    
    report_best_vrp(current_routes)
    
    # Intra-route 2-opt
    for r_idx in range(truck_count):
        route = current_routes[r_idx]
        improved = True
        max_iters = len(route) * 10
        it = 0
        while improved and it < max_iters:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = compute_route_distance(new_route)
                    if new_dist < route_distances[r_idx]:
                        route_distances[r_idx] = new_dist
                        current_routes[r_idx] = new_route
                        improved = True
                        new_max = max(route_distances)
                        if new_max < current_max:
                            current_max = new_max
                            report_best_vrp(current_routes)
                        break
                if improved:
                    break
            it += 1
    
    # Inter-route relocate from longest route
    improved = True
    it = 0
    while improved and it < n:
        improved = False
        max_route_idx = max(range(truck_count), key=lambda r: route_distances[r])
        route_max = current_routes[max_route_idx]
        if len(route_max) <= 2:
            break
        # Try removing each interior customer from longest route and insert into other routes
        for idx in range(1, len(route_max)-1):
            cust = route_max[idx]
            # new route after removal without cust
            new_route_max = route_max[:idx] + route_max[idx+1:]
            new_dist_max = compute_route_distance(new_route_max)
            for r_idx in range(truck_count):
                if r_idx == max_route_idx:
                    continue
                route = current_routes[r_idx]
                for i in range(1, len(route)):
                    new_route_other = route[:i] + [cust] + route[i:]
                    new_dist_other = compute_route_distance(new_route_other)
                    other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:max_route_idx] + route_distances[max_route_idx+1:])
                    new_max = max(other_max, new_dist_max, new_dist_other)
                    if new_max < current_max:
                        # Apply move
                        current_routes[max_route_idx] = new_route_max
                        route_distances[max_route_idx] = new_dist_max
                        current_routes[r_idx] = new_route_other
                        route_distances[r_idx] = new_dist_other
                        current_max = new_max
                        improved = True
                        report_best_vrp(current_routes)
                        break
                if improved:
                    break
            if improved:
                break
        it += 1
    
    # Inter-route swap (same as parent)
    improved = True
    max_iters = n * n
    it = 0
    while improved and it < max_iters:
        improved = False
        for r1 in range(truck_count):
            for r2 in range(r1+1, truck_count):
                route1 = current_routes[r1]
                route2 = current_routes[r2]
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        new1 = route1[:i] + [route2[j]] + route1[i+1:]
                        new2 = route2[:j] + [route1[i]] + route2[j+1:]
                        new_dist1 = compute_route_distance(new1)
                        new_dist2 = compute_route_distance(new2)
                        other_max = max(route_distances[:r1] + route_distances[r1+1:r2] + route_distances[r2+1:])
                        new_max = max(other_max, new_dist1, new_dist2)
                        if new_max < current_max:
                            current_routes[r1] = new1
                            current_routes[r2] = new2
                            route_distances[r1] = new_dist1
                            route_distances[r2] = new_dist2
                            current_max = new_max
                            improved = True
                            report_best_vrp(current_routes)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        it += 1
    return current_routes