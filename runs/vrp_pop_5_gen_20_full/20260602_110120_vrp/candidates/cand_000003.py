import numpy as np
from itertools import permutations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_distances = [2 * distance_matrix[0, 0] for _ in range(truck_count)]  # actually 0
    # Correct route distance: sum of edges along route
    def compute_route_distance(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    for r in range(truck_count):
        route_distances[r] = compute_route_distance(routes[r])
    unassigned = set(range(1, n))
    
    def best_max(customer):
        best_val = float('inf')
        best_pos = None
        second_val = float('inf')
        for r_idx, route in enumerate(routes):
            curr_dist = route_distances[r_idx]
            # positions from 1 to len(route)-1
            for i in range(1, len(route)):
                # new distance for route if insert at i
                new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], customer] + distance_matrix[customer, route[i]]
                # new max distance
                other_max = max(route_distances[:r_idx] + route_distances[r_idx+1:], default=0)
                cand_max = max(new_dist, other_max)
                if cand_max < best_val:
                    second_val = best_val
                    best_val = cand_max
                    best_pos = (r_idx, i)
                elif cand_max < second_val and cand_max != best_val:
                    second_val = cand_max
        return best_val, second_val, best_pos
    
    while unassigned:
        # compute regret for each unassigned
        regrets = []
        for c in unassigned:
            best_val, second_val, _ = best_max(c)
            regret = second_val - best_val if second_val != float('inf') else 0
            regrets.append((regret, c, best_val, second_val))
        # select customer with largest regret, tie by smallest customer index
        regrets.sort(key=lambda x: (-x[0], x[1]))
        selected = regrets[0][1]
        # get best insertion for selected again
        best_val, _, best_pos = best_max(selected)
        r_idx, i = best_pos
        route = routes[r_idx]
        # update route
        route.insert(i, selected)
        route_distances[r_idx] = compute_route_distance(route)
        unassigned.remove(selected)
    
    # improvement phase
    current_routes = [list(r) for r in routes]
    current_max = max(route_distances)
    # report initial
    def report_best_vrp(routes):
        pass  # placeholder, actual call will be inserted by evaluator
    # intra-route 2-opt for each route
    for r_idx in range(truck_count):
        route = current_routes[r_idx]
        improved = True
        max_iters = len(route) * 10  # finite
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
    # inter-route swap
    improved = True
    max_iters = n * n  # finite
    it = 0
    while improved and it < max_iters:
        improved = False
        for r1 in range(truck_count):
            for r2 in range(r1+1, truck_count):
                route1 = current_routes[r1]
                route2 = current_routes[r2]
                for i in range(1, len(route1)-1):
                    for j in range(1, len(route2)-1):
                        # swap customers
                        new1 = route1[:i] + [route2[j]] + route1[i+1:]
                        new2 = route2[:j] + [route1[i]] + route2[j+1:]
                        new_dist1 = compute_route_distance(new1)
                        new_dist2 = compute_route_distance(new2)
                        new_max = max(max(route_distances[:r1] + route_distances[r1+1:r2] + route_distances[r2+1:]), new_dist1, new_dist2)
                        if new_max < current_max:
                            # update
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