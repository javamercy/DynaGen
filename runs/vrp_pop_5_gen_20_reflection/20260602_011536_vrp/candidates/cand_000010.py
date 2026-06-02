import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0] * truck_count
    customers = list(range(1, n))
    for c in customers:
        best_new_max = float('inf')
        best_r = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            current_dist = route_dists[r]
            for i in range(len(route) - 1):
                a = route[i]
                b = route[i+1]
                increase = distance_matrix[a][c] + distance_matrix[c][b] - distance_matrix[a][b]
                new_dist = current_dist + increase
                other_max = max(route_dists[:r] + route_dists[r+1:])
                new_max = max(other_max, new_dist)
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_r = r
                    best_pos = i
        routes[best_r].insert(best_pos + 1, c)
        route_dists[best_r] += distance_matrix[routes[best_r][best_pos]][c] + distance_matrix[c][routes[best_r][best_pos+2]] - distance_matrix[routes[best_r][best_pos]][routes[best_r][best_pos+2]]
    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    report_best_vrp(best_routes)
    # Single pass 2-opt per route
    for r in range(truck_count):
        route = routes[r]
        improved = True
        for i in range(1, len(route) - 2):
            for k in range(i+1, len(route) - 1):
                new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
                new_dist = 0
                for j in range(len(new_route) - 1):
                    new_dist += distance_matrix[new_route[j]][new_route[j+1]]
                if new_dist < route_dists[r]:
                    routes[r] = new_route
                    route_dists[r] = new_dist
                    improved = True
                    current_max = max(route_dists)
                    if current_max < best_max:
                        best_max = current_max
                        best_routes = [list(r2) for r2 in routes]
                        report_best_vrp(best_routes)
                    break
            if improved:
                break
    final_max = max(route_dists)
    if final_max < best_max:
        best_routes = [list(r) for r in routes]
    return best_routes