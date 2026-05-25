import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_new_max(routes, route_dists, cust, r, pos):
        route = routes[r]
        inc = (distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]])
        new_dist = route_dists[r] + inc
        other_max = max(route_dists[i] for i in range(truck_count) if i != r)
        return max(new_dist, other_max)

    def insert_best(routes, route_dists, cust, best_r, best_pos):
        routes[best_r].insert(best_pos, cust)
        route_dists[best_r] = route_distance(routes[best_r])

    # Construction: farthest-first
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0 for _ in range(truck_count)]
    customers = list(range(1, n))
    customers.sort(key=lambda c: distance_matrix[0][c], reverse=True)
    for cust in customers:
        best_max = float('inf')
        best_r = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            for pos in range(1, len(route)):
                new_max = compute_new_max(routes, route_dists, cust, r, pos)
                if new_max < best_max - 1e-9:
                    best_max = new_max
                    best_r = r
                    best_pos = pos
                elif abs(new_max - best_max) < 1e-9:
                    if r < best_r or (r == best_r and pos < best_pos):
                        best_r = r
                        best_pos = pos
        insert_best(routes, route_dists, cust, best_r, best_pos)

    best_routes = [list(r) for r in routes]
    best_max = max(route_dists)
    try:
        report_best_vrp(best_routes)
    except NameError:
        pass

    # Local search: relocate and swap only
    max_iters = 10 * n * truck_count
    for _ in range(max_iters):
        improved = False
        # Relocate
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for idx in range(1, len(route1)-1):
                cust = route1[idx]
                new_route1 = route1[:idx] + route1[idx+1:]
                new_dist1 = route_distance(new_route1)
                for r2 in range(truck_count):
                    if r2 == r1:
                        continue
                    route2 = routes[r2]
                    for pos in range(1, len(route2)):
                        inc = (distance_matrix[route2[pos-1], cust] + distance_matrix[cust, route2[pos]] - distance_matrix[route2[pos-1], route2[pos]])
                        new_dist2 = route_dists[r2] + inc
                        other_dists = [route_dists[i] for i in range(truck_count) if i not in (r1, r2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        if new_max < best_max - 1e-9:
                            routes[r1] = new_route1
                            route_dists[r1] = new_dist1
                            routes[r2].insert(pos, cust)
                            route_dists[r2] = new_dist2
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Swap
        for r1 in range(truck_count):
            route1 = routes[r1]
            if len(route1) <= 2:
                continue
            for idx1 in range(1, len(route1)-1):
                cust1 = route1[idx1]
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 2:
                        continue
                    for idx2 in range(1, len(route2)-1):
                        cust2 = route2[idx2]
                        new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                        new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                        new_dist1 = route_distance(new_route1)
                        new_dist2 = route_distance(new_route2)
                        other_dists = [route_dists[i] for i in range(truck_count) if i not in (r1, r2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        if new_max < best_max - 1e-9:
                            routes[r1] = new_route1
                            routes[r2] = new_route2
                            route_dists[r1] = new_dist1
                            route_dists[r2] = new_dist2
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            improved = True
                            try:
                                report_best_vrp(best_routes)
                            except NameError:
                                pass
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    # Ensure empty routes
    for r in range(truck_count):
        if len(best_routes[r]) == 0:
            best_routes[r] = [0, 0]
    return best_routes