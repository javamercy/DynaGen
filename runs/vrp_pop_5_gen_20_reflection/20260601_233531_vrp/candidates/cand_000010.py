import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    routes = [[0, c, 0] for c in range(1, n)]

    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    while len(routes) > truck_count:
        best_pair = None
        best_max = float('inf')
        best_total = float('inf')
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                merged = routes[i][:-1] + routes[j][1:]
                dist = route_dist(merged)
                max_dist = dist
                for k in range(len(routes)):
                    if k != i and k != j:
                        dk = route_dist(routes[k])
                        if dk > max_dist:
                            max_dist = dk
                if (max_dist < best_max) or (max_dist == best_max and dist < best_total):
                    best_max = max_dist
                    best_total = dist
                    best_pair = (i, j, merged)
        i, j, merged = best_pair
        routes[i] = merged
        del routes[j]
    while len(routes) < truck_count:
        routes.append([0, 0])
    report_best_vrp(routes)
    return routes