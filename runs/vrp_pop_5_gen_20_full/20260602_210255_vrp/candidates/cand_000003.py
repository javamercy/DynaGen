import numpy as np
import itertools

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = set(customers)
    
    # Regret-2 insertion construction
    while unvisited:
        best_node = None
        best_regret = -1
        best_route_idx = None
        best_pos = None
        for node in unvisited:
            costs = []
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    # insertion between route[pos-1] and route[pos]
                    prev = route[pos-1]
                    next_ = route[pos]
                    new_cost = distance_matrix[prev, node] + distance_matrix[node, next_] - distance_matrix[prev, next_]
                    costs.append((new_cost, r_idx, pos))
            costs.sort(key=lambda x: x[0])
            if len(costs) == 1:
                regret = 0
                selected = costs[0]
            else:
                regret = costs[1][0] - costs[0][0]
                selected = costs[0]
            if regret > best_regret or (regret == best_regret and node < best_node if best_node is not None else True):
                best_regret = regret
                best_node = node
                best_route_idx, best_pos = selected[1], selected[2]
        # insert best_node
        route = routes[best_route_idx]
        route.insert(best_pos, best_node)
        unvisited.remove(best_node)
    
    best_routes = [route.copy() for route in routes]
    best_max_dist = max(route_distance(route, distance_matrix) for route in best_routes)
    # report initial
    report_best_vrp(best_routes)
    
    def route_distance(route, dist):
        d = 0
        for i in range(len(route)-1):
            d += dist[route[i], route[i+1]]
        return d
    
    # Improvement: iterate limited number of times
    max_iter = n * truck_count * 2
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        # Intra-route 2-opt
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            old_dist = route_distance(route, distance_matrix)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = route_distance(new_route, distance_matrix)
                    # accept if reduces max distance or reduces total without increasing max
                    current_max = max(route_distance(r, distance_matrix) for r in routes)
                    # compute new max after change
                    other_routes = [routes[k] for k in range(truck_count) if k != r_idx]
                    new_max = max(new_dist, max(route_distance(r, distance_matrix) for r in other_routes))
                    if new_max < current_max or (new_max == current_max and new_dist < old_dist):
                        routes[r_idx] = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            # update best
            current_max = max(route_distance(r, distance_matrix) for r in routes)
            if current_max < best_max_dist:
                best_max_dist = current_max
                best_routes = [r.copy() for r in routes]
                report_best_vrp(best_routes)
            continue
        # Inter-route relocate: move a customer from longest route to another
        # identify route with max distance
        dists = [route_distance(r, distance_matrix) for r in routes]
        max_idx = np.argmax(dists)
        max_route = routes[max_idx]
        if len(max_route) <= 2:
            break
        # try moving each interior node to other routes
        for m in range(1, len(max_route)-1):
            node = max_route[m]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                # try all insertion positions
                for pos in range(1, len(other_route)):
                    # compute new max
                    new_max_route = max_route[:m] + max_route[m+1:]
                    new_max_dist = route_distance(new_max_route, distance_matrix)
                    new_other_route = other_route[:pos] + [node] + other_route[pos:]
                    new_other_dist = route_distance(new_other_route, distance_matrix)
                    # compute overall max
                    other_route_dists = [route_distance(routes[k], distance_matrix) for k in range(truck_count) if k not in (max_idx, other_idx)]
                    new_max = max(new_max_dist, new_other_dist, *other_route_dists)
                    current_max = max(dists)
                    if new_max < current_max:
                        # perform move
                        routes[max_idx] = new_max_route
                        routes[other_idx] = new_other_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            current_max = max(route_distance(r, distance_matrix) for r in routes)
            if current_max < best_max_dist:
                best_max_dist = current_max
                best_routes = [r.copy() for r in routes]
                report_best_vrp(best_routes)
    return best_routes