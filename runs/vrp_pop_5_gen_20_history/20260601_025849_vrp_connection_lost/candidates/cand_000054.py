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

    # Regret-3 construction
    while unassigned:
        best_infos = {}
        for c in unassigned:
            costs = []
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    cost = dist[route[i]][c] + dist[c][route[i+1]] - dist[route[i]][route[i+1]]
                    costs.append((cost, r_idx, i+1))
            costs.sort(key=lambda x: x[0])
            top3 = costs[:3]
            if len(top3) < 3:
                regret = float('inf')
            else:
                c1, c2, c3 = top3[0][0], top3[1][0], top3[2][0]
                regret = (c2 + c3) - 2 * c1
            best_r = top3[0][1]
            best_p = top3[0][2]
            new_route = routes[best_r][:best_p] + [c] + routes[best_r][best_p:]
            new_route_dist = route_dist(new_route)
            other_max = 0.0
            if truck_count > 1:
                other_max = max(route_dist(r) for i, r in enumerate(routes) if i != best_r)
            new_max = max(new_route_dist, other_max)
            best_infos[c] = (-regret, new_max, c, best_r, best_p)

        candidates = list(best_infos.values())
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_c, chosen_r, chosen_p = candidates[0]
        routes[chosen_r].insert(chosen_p, chosen_c)
        unassigned.remove(chosen_c)

    report_best_vrp(routes)
    best_routes = [list(r) for r in routes]
    best_max = max_dist(best_routes)

    # Improvement with adaptive acceptance threshold
    max_iter = n * n
    stagnation_limit = n * truck_count  # increased for more exploration
    consecutive_no_improvement = 0
    initial_threshold = 0.1
    threshold = initial_threshold

    for iteration in range(max_iter):
        improved = False
        current_max = max_dist(routes)
        longest_indices = [i for i, r in enumerate(routes) if route_dist(r) == current_max]
        if not longest_indices:
            break
        r_idx = longest_indices[0]
        route = routes[r_idx]

        # Decay threshold linearly
        threshold = initial_threshold * (1 - iteration / max_iter)

        # Relocate from longest route
        for pos in range(1, len(route) - 1):
            cust = route[pos]
            for other_idx, other_route in enumerate(routes):
                if other_idx == r_idx:
                    continue
                for other_pos in range(1, len(other_route)):
                    new_self = route[:pos] + route[pos+1:]
                    new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                    new_routes = [list(r) for r in routes]
                    new_routes[r_idx] = new_self
                    new_routes[other_idx] = new_other
                    new_max = max_dist(new_routes)
                    if new_max < best_max * (1 + threshold):
                        best_max = min(best_max, new_max)
                        best_routes = [list(r) for r in new_routes]
                        routes = new_routes
                        improved = True
                        report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            consecutive_no_improvement = 0
            continue

        # Inter-route swap
        for pos1 in range(1, len(routes[r_idx]) - 1):
            cust1 = routes[r_idx][pos1]
            for other_idx, other_route in enumerate(routes):
                if other_idx == r_idx:
                    continue
                for pos2 in range(1, len(other_route) - 1):
                    cust2 = other_route[pos2]
                    new_route1 = routes[r_idx][:pos1] + [cust2] + routes[r_idx][pos1+1:]
                    new_route2 = other_route[:pos2] + [cust1] + other_route[pos2+1:]
                    new_routes = [list(r) for r in routes]
                    new_routes[r_idx] = new_route1
                    new_routes[other_idx] = new_route2
                    new_max = max_dist(new_routes)
                    if new_max < best_max * (1 + threshold):
                        best_max = min(best_max, new_max)
                        best_routes = [list(r) for r in new_routes]
                        routes = new_routes
                        improved = True
                        report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            consecutive_no_improvement = 0
            continue

        # 2-opt on each route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_dist(new_route) < route_dist(route):
                        routes[r_idx] = new_route
                        improved = True
                        current_max = max_dist(routes)
                        if current_max < best_max:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
            if consecutive_no_improvement >= stagnation_limit:
                break

    # Ensure exactly truck_count routes
    final_routes = []
    for route in best_routes:
        if len(route) == 2:
            final_routes.append([0, 0])
        else:
            final_routes.append([0] + route[1:-1] + [0])
    return final_routes