import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    unassigned = set(range(1, n))
    
    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def max_route_distance(routes):
        return max(route_distance(r) for r in routes)
    
    def insert_customer(route, pos, cust):
        return route[:pos] + [cust] + route[pos:]
    
    # Regret-2 construction with min-max cost
    while unassigned:
        best_cust = None
        best_regret = -1.0
        best_route_idx = -1
        best_pos = -1
        best_new_max = float('inf')
        for cust in sorted(unassigned):
            best_cost = float('inf')
            second_best_cost = float('inf')
            best_r = -1
            best_p = -1
            best_new_max_cust = float('inf')
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    new_route = insert_customer(route, pos, cust)
                    new_dist = route_distance(new_route)
                    other_max = 0.0
                    if truck_count > 1:
                        other_max = max(route_distance(routes[i]) for i in range(truck_count) if i != r_idx)
                    new_max = max(new_dist, other_max)
                    if new_max < best_cost:
                        second_best_cost = best_cost
                        best_cost = new_max
                        best_r = r_idx
                        best_p = pos
                        best_new_max_cust = new_max
                    elif new_max < second_best_cost:
                        second_best_cost = new_max
            if best_cost == float('inf'):
                continue
            regret = second_best_cost - best_cost
            # Tie-breaking: higher regret, then lower customer index, then lower new_max, then lower route index, then lower position (implicitly by order of iteration)
            if regret > best_regret or (regret == best_regret and cust < best_cust) or (regret == best_regret and cust == best_cust and best_new_max_cust < best_new_max):
                best_regret = regret
                best_cust = cust
                best_route_idx = best_r
                best_pos = best_p
                best_new_max = best_new_max_cust
        if best_cust is None:
            break
        routes[best_route_idx] = insert_customer(routes[best_route_idx], best_pos, best_cust)
        unassigned.remove(best_cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max_route_distance(best_routes)
    report_best_vrp(best_routes)
    
    # Improvement with stagnation detection
    max_iter = n * n
    stagnation_limit = max(10, (n-1) // 10)
    no_improve = 0
    
    for _ in range(max_iter):
        improved = False
        # Relocate from longest to shortest
        dists = [route_distance(r) for r in routes]
        current_max = max(dists)
        longest_indices = [i for i, d in enumerate(dists) if d == current_max]
        if not longest_indices:
            break
        r_idx = longest_indices[0]
        min_idx = min(range(len(dists)), key=lambda i: dists[i])
        if dists[r_idx] == dists[min_idx]:
            # all equal, try other moves
            pass
        route_long = routes[r_idx]
        route_short = routes[min_idx]
        best_delta = 0
        best_move = None
        for pos in range(1, len(route_long)-1):
            cust = route_long[pos]
            new_long = route_long[:pos] + route_long[pos+1:]
            if len(new_long) == 2:
                new_long = [0, 0]
            for ins in range(1, len(route_short)):
                new_short = route_short[:ins] + [cust] + route_short[ins:]
                d_long = route_distance(new_long)
                d_short = route_distance(new_short)
                new_max = max(d_long, d_short, max(dists[k] for k in range(len(dists)) if k not in (r_idx, min_idx)))
                if new_max < best_max:
                    delta = best_max - new_max
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (r_idx, min_idx, pos, ins, new_long, new_short)
        if best_move:
            r_idx, min_idx, pos, ins, new_long, new_short = best_move
            routes[r_idx] = new_long
            routes[min_idx] = new_short
            new_max = max_route_distance(routes)
            if new_max < best_max:
                best_max = new_max
                best_routes = [list(r) for r in routes]
                report_best_vrp(best_routes)
                improved = True
                no_improve = 0
                continue
        # Inter-route swap
        if not improved:
            for r1 in range(truck_count):
                if len(routes[r1]) <= 3:
                    continue
                for pos1 in range(1, len(routes[r1])-1):
                    cust1 = routes[r1][pos1]
                    for r2 in range(r1+1, truck_count):
                        if len(routes[r2]) <= 3:
                            continue
                        for pos2 in range(1, len(routes[r2])-1):
                            cust2 = routes[r2][pos2]
                            new_route1 = routes[r1][:pos1] + [cust2] + routes[r1][pos1+1:]
                            new_route2 = routes[r2][:pos2] + [cust1] + routes[r2][pos2+1:]
                            new_routes = [list(r) for r in routes]
                            new_routes[r1] = new_route1
                            new_routes[r2] = new_route2
                            new_max = max_route_distance(new_routes)
                            if new_max < best_max:
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
        # 2-opt within each route
        for r_idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route):
                        routes[r_idx] = new_route
                        improved = True
                        current_max = max_route_distance(routes)
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
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= stagnation_limit:
                break
    
    final_routes = []
    for r in best_routes:
        if len(r) == 2:
            final_routes.append([0, 0])
        else:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
            final_routes.append(r)
    return final_routes