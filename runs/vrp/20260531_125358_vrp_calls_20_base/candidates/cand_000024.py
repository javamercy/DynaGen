import random
import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = []
        for i in range(truck_count):
            if i < len(customers):
                routes.append([0, customers[i], 0])
            else:
                routes.append([0, 0])
        return routes

    def route_length(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def split_permutation(perm):
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        for cust in perm:
            best_route = -1
            best_new_max = float('inf')
            best_total = float('inf')
            for r in range(truck_count):
                new_route = routes[r][:-1] + [cust] + [0]
                new_len = route_length(new_route)
                other_max = 0.0
                for rr in range(truck_count):
                    if rr != r:
                        other_max = max(other_max, route_dists[rr])
                new_max = max(new_len, other_max)
                total = sum(route_dists) - route_dists[r] + new_len
                if new_max < best_new_max or (new_max == best_new_max and total < best_total):
                    best_new_max = new_max
                    best_total = total
                    best_route = r
            routes[best_route] = routes[best_route][:-1] + [cust] + [0]
            route_dists[best_route] = route_length(routes[best_route])
        max_dist = max(route_dists)
        return routes, max_dist

    random.seed(0)
    best_routes = None
    best_max = float('inf')
    num_restarts = min(5, max(1, n // 20))
    for _ in range(num_restarts):
        perm = customers[:]
        random.shuffle(perm)
        current_routes, current_max = split_permutation(perm)
        if current_max < best_max:
            best_max = current_max
            best_routes = [r[:] for r in current_routes]
            report_best_vrp(best_routes)
        # Local search on permutation
        improved = True
        max_iter = n * 5
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            # Swap moves
            for i in range(len(perm)):
                for j in range(i+1, len(perm)):
                    new_perm = perm[:]
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    routes, max_dist = split_permutation(new_perm)
                    if max_dist < best_max:
                        best_max = max_dist
                        best_routes = [r[:] for r in routes]
                        perm = new_perm
                        improved = True
                        report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                it = 0
                continue
            # Inversion moves
            for i in range(len(perm)):
                for j in range(i+1, len(perm)):
                    new_perm = perm[:i] + perm[i:j+1][::-1] + perm[j+1:]
                    routes, max_dist = split_permutation(new_perm)
                    if max_dist < best_max:
                        best_max = max_dist
                        best_routes = [r[:] for r in routes]
                        perm = new_perm
                        improved = True
                        report_best_vrp(best_routes)
                        break
                if improved:
                    break
    return best_routes