import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    dist = distance_matrix
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))

    def route_dist(route):
        d = 0.0
        for a, b in zip(route, route[1:]):
            d += dist[a][b]
        return d

    def max_dist(routes):
        return max(route_dist(r) for r in routes)

    # Regret-2 construction (from cand_000035)
    while unassigned:
        best_info = {}
        for c in unassigned:
            best = float('inf')
            second = float('inf')
            best_r = -1
            best_p = -1
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                    if cost < best:
                        second = best
                        best = cost
                        best_r = r_idx
                        best_p = i + 1
                    elif cost < second:
                        second = cost
            best_info[c] = (best, second, best_r, best_p)

        candidates = []
        for c, (best, second, r_idx, pos) in best_info.items():
            regret = second - best if second != float('inf') else float('inf')
            new_route = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
            new_route_dist = route_dist(new_route)
            other_max = 0.0
            if truck_count > 1:
                other_max = max(route_dist(r) for i, r in enumerate(routes) if i != r_idx)
            new_max = max(new_route_dist, other_max)
            candidates.append((-regret, new_max, c, r_idx, pos))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_c, chosen_r, chosen_p = candidates[0]
        routes[chosen_r].insert(chosen_p, chosen_c)
        unassigned.remove(chosen_c)

    report_best_vrp(routes)
    best_routes = [list(r) for r in routes]
    best_max = max_dist(best_routes)

    # Best-improvement local search
    max_iter = n * n
    for _ in range(max_iter):
        best_move = None
        best_new_max = float('inf')
        best_new_routes = None

        # Relocate moves: from each route, each customer, to every other route at every position
        for r_idx_src, route_src in enumerate(routes):
            for pos_src in range(1, len(route_src) - 1):
                cust = route_src[pos_src]
                for r_idx_dst in range(truck_count):
                    if r_idx_dst == r_idx_src:
                        continue
                    route_dst = routes[r_idx_dst]
                    for pos_dst in range(1, len(route_dst)):
                        new_self = route_src[:pos_src] + route_src[pos_src+1:]
                        new_other = route_dst[:pos_dst] + [cust] + route_dst[pos_dst:]
                        new_routes = [list(r) for r in routes]
                        new_routes[r_idx_src] = new_self
                        new_routes[r_idx_dst] = new_other
                        new_max = max_dist(new_routes)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_new_routes = new_routes
                            best_move = 'relocate'

        # Swap moves: between two routes, swap customers
        for r_idx1, route1 in enumerate(routes):
            for pos1 in range(1, len(route1) - 1):
                cust1 = route1[pos1]
                for r_idx2, route2 in enumerate(routes):
                    if r_idx2 <= r_idx1:
                        continue
                    for pos2 in range(1, len(route2) - 1):
                        cust2 = route2[pos2]
                        new_route1 = route1[:pos1] + [cust2] + route1[pos1+1:]
                        new_route2 = route2[:pos2] + [cust1] + route2[pos2+1:]
                        new_routes = [list(r) for r in routes]
                        new_routes[r_idx1] = new_route1
                        new_routes[r_idx2] = new_route2
                        new_max = max_dist(new_routes)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_new_routes = new_routes
                            best_move = 'swap'

        # 2-opt moves: reverse subsequence within a route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_routes = [list(r) for r in routes]
                    new_routes[r_idx] = new_route
                    new_max = max_dist(new_routes)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_new_routes = new_routes
                        best_move = '2opt'

        if best_new_max < best_max:
            best_max = best_new_max
            best_routes = best_new_routes
            routes = best_new_routes
            report_best_vrp(best_routes)
        else:
            break

    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + route[1:-1] + [0])
    return final_routes