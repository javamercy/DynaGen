import numpy as np
from itertools import permutations

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
            if len(data) >= 3:
                regret = (data[1][0] - data[0][0]) + (data[2][0] - data[0][0])
            elif len(data) == 2:
                regret = data[1][0] - data[0][0]
            else:
                regret = 0.0
            regrets.append((regret, c, data[0][1]))
        regrets.sort(key=lambda x: (-x[0], x[1]))
        selected = regrets[0][1]
        best_pos = regrets[0][2]
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
    
    # Inter-route swap
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
                        new_max = max(max(route_distances[:r1] + route_distances[r1+1:r2] + route_distances[r2+1:]), new_dist1, new_dist2)
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