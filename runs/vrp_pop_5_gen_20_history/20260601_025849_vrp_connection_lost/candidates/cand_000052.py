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

    # Regret-3 construction with min-max tie-breaking
    while unassigned:
        best_cust = None
        best_regret = -float('inf')
        best_new_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for cust in sorted(unassigned):
            costs = []
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    cost = dist[route[i]][cust] + dist[cust][route[i+1]] - dist[route[i]][route[i+1]]
                    new_route = route[:i+1] + [cust] + route[i+1:]
                    new_dist = route_dist(new_route)
                    other_max = 0.0
                    if truck_count > 1:
                        other_max = max(route_dist(routes[j]) for j in range(truck_count) if j != r_idx)
                    new_max = max(new_dist, other_max)
                    costs.append((cost, new_max, r_idx, i+1))
            costs.sort(key=lambda x: x[0])
            top3 = costs[:3]
            if len(top3) < 3:
                regret = float('inf')
            else:
                c1, c2, c3 = top3[0][0], top3[1][0], top3[2][0]
                regret = (c2 + c3) - 2 * c1
            # Use the best option's new_max and route/pos
            best_cost = top3[0]
            new_max = best_cost[1]
            r_idx = best_cost[2]
            pos = best_cost[3]
            # Tie-breaking: higher regret, then lower new_max, then lower customer index
            if regret > best_regret or (regret == best_regret and new_max < best_new_max) or (regret == best_regret and abs(new_max - best_new_max) < 1e-12 and cust < best_cust):
                best_regret = regret
                best_new_max = new_max
                best_cust = cust
                best_route_idx = r_idx
                best_pos = pos
        if best_cust is None:
            break
        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)

    best_routes = [list(r) for r in routes]
    best_max = max_dist(best_routes)
    report_best_vrp(best_routes)

    # Improvement with adaptive stagnation
    max_iter = n * n
    stagnation_limit = max(10, (n - 1) // (truck_count + 1))
    no_improve = 0

    for _ in range(max_iter):
        improved = False
        current_max = max_dist(routes)
        longest_indices = [i for i, r in enumerate(routes) if route_dist(r) == current_max]
        if not longest_indices:
            break
        r_idx = longest_indices[0]
        # Relocate from longest route to shortest route
        dists = [route_dist(r) for r in routes]
        min_idx = min(range(len(dists)), key=lambda i: dists[i])
        if dists[r_idx] > dists[min_idx]:
            route_long = routes[r_idx]
            route_short = routes[min_idx]
            best_delta = 0
            best_move = None
            for pos in range(1, len(route_long) - 1):
                cust = route_long[pos]
                new_long = route_long[:pos] + route_long[pos+1:]
                if len(new_long) == 2:
                    new_long = [0, 0]
                for ins in range(1, len(route_short)):
                    new_short = route_short[:ins] + [cust] + route_short[ins:]
                    new_routes = [list(r) for r in routes]
                    new_routes[r_idx] = new_long
                    new_routes[min_idx] = new_short
                    new_max = max_dist(new_routes)
                    if new_max < best_max - 1e-12:
                        delta = best_max - new_max
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (r_idx, min_idx, pos, ins, new_long, new_short)
            if best_move is not None:
                r_idx, min_idx, pos, ins, new_long, new_short = best_move
                routes[r_idx] = new_long
                routes[min_idx] = new_short
                new_max = max_dist(routes)
                if new_max < best_max - 1e-12:
                    best_max = new_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                    improved = True
                    no_improve = 0
                    continue
        # If no relocate improvement, try inter-route swap
        if not improved:
            for r1 in range(truck_count):
                if len(routes[r1]) <= 3:
                    continue
                for pos1 in range(1, len(routes[r1]) - 1):
                    cust1 = routes[r1][pos1]
                    for r2 in range(r1 + 1, truck_count):
                        if len(routes[r2]) <= 3:
                            continue
                        for pos2 in range(1, len(routes[r2]) - 1):
                            cust2 = routes[r2][pos2]
                            new_route1 = routes[r1][:pos1] + [cust2] + routes[r1][pos1+1:]
                            new_route2 = routes[r2][:pos2] + [cust1] + routes[r2][pos2+1:]
                            new_routes = [list(r) for r in routes]
                            new_routes[r1] = new_route1
                            new_routes[r2] = new_route2
                            new_max = max_dist(new_routes)
                            if new_max < best_max - 1e-12:
                                best_max = new_max
                                best_routes = new_routes
                                routes = new_routes
                                report_best_vrp(best_routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        if improved:
            continue
        # If still no improvement, try 2-opt within each route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_dist(new_route) < route_dist(route) - 1e-12:
                        routes[r_idx] = new_route
                        improved = True
                        current_max = max_dist(routes)
                        if current_max < best_max - 1e-12:
                            best_max = current_max
                            best_routes = [list(r) for r in routes]
                            report_best_vrp(best_routes)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= stagnation_limit:
                break

    # Final cleanup: ensure exactly truck_count routes with proper format
    final_routes = []
    for route in best_routes:
        if len(route) == 2 and route[0] == route[1] == 0:
            final_routes.append([0, 0])
        else:
            final_route = [0]
            for c in route:
                if c != 0:
                    final_route.append(c)
            final_route.append(0)
            final_routes.append(final_route)
    return final_routes