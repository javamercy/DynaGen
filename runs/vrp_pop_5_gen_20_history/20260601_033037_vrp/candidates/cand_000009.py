import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    routes = [[] for _ in range(truck_count)]
    unassigned = sorted(customers)
    total_cust = len(unassigned)

    def route_distance(route):
        if not route:
            return 0.0
        d = distance_matrix[0, route[0]] + distance_matrix[route[-1], 0]
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def route_max():
        return max(route_distance(r) for r in routes)

    current_max = None

    def report_if_better():
        nonlocal current_max
        maxd = route_max()
        if current_max is None or maxd < current_max:
            current_max = maxd
            full_routes = [[0] + r + [0] for r in routes]
            report_best_vrp(full_routes)

    # Adaptive regret insertion with blend
    assigned = 0
    while unassigned:
        alpha = 1.0 - 0.5 * (assigned / total_cust)  # decays from 1.0 to 0.5
        best_score = -1e100
        best_cust = None
        best_route_idx = None
        best_pos = None
        best_delta = None

        for cust in sorted(unassigned):
            route_best = []
            for r_idx, route in enumerate(routes):
                if not route:
                    delta = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    route_best.append((delta, r_idx, 0))
                else:
                    min_delta = float('inf')
                    best_p = None
                    for pos in range(len(route)+1):
                        if pos == 0:
                            prev = 0
                            next = route[0]
                        elif pos == len(route):
                            prev = route[-1]
                            next = 0
                        else:
                            prev = route[pos-1]
                            next = route[pos]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, next] - distance_matrix[prev, next]
                        if delta < min_delta:
                            min_delta = delta
                            best_p = pos
                    route_best.append((min_delta, r_idx, best_p))
            route_best.sort(key=lambda x: x[0])
            best = route_best[0][0]
            second = route_best[1][0] if len(route_best) > 1 else best
            regret = second - best
            score = regret * alpha + (1 - alpha) * (-best)  # negative best because lower cost is better
            if score > best_score:
                best_score = score
                best_cust = cust
                best_delta = best
                best_route_idx = route_best[0][1]
                best_pos = route_best[0][2]

        route = routes[best_route_idx]
        route.insert(best_pos, best_cust)
        unassigned.remove(best_cust)
        assigned += 1
        report_if_better()

    # Improvement: relocate from longest route to others
    for _ in range(n):
        max_dist = 0
        max_idx = 0
        for i, r in enumerate(routes):
            d = route_distance(r)
            if d > max_dist:
                max_dist = d
                max_idx = i
        route_to_shrink = routes[max_idx]
        if len(route_to_shrink) == 0:
            break
        moved = False
        for cust in list(route_to_shrink):
            if moved:
                break
            pos = route_to_shrink.index(cust)
            if len(route_to_shrink) == 1:
                delta_rem = -distance_matrix[0, cust] - distance_matrix[cust, 0]
            else:
                if pos == 0:
                    prev = 0
                    next = route_to_shrink[1]
                elif pos == len(route_to_shrink)-1:
                    prev = route_to_shrink[-2]
                    next = 0
                else:
                    prev = route_to_shrink[pos-1]
                    next = route_to_shrink[pos+1]
                delta_rem = distance_matrix[prev, next] - distance_matrix[prev, cust] - distance_matrix[cust, next]
            new_route_long = route_to_shrink[:pos] + route_to_shrink[pos+1:]
            new_dist_long = route_distance(new_route_long)
            for r_idx, other_route in enumerate(routes):
                if r_idx == max_idx:
                    continue
                best_insert_delta = float('inf')
                best_pos_other = None
                if not other_route:
                    delta_ins = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    best_insert_delta = delta_ins
                    best_pos_other = 0
                else:
                    for p in range(len(other_route)+1):
                        if p == 0:
                            prev = 0
                            next = other_route[0]
                        elif p == len(other_route):
                            prev = other_route[-1]
                            next = 0
                        else:
                            prev = other_route[p-1]
                            next = other_route[p]
                        delta = distance_matrix[prev, cust] + distance_matrix[cust, next] - distance_matrix[prev, next]
                        if delta < best_insert_delta:
                            best_insert_delta = delta
                            best_pos_other = p
                new_other_route = other_route[:best_pos_other] + [cust] + other_route[best_pos_other:]
                new_dist_other = route_distance(new_other_route)
                new_max = max(new_dist_long, new_dist_other)
                for r2_idx, r2 in enumerate(routes):
                    if r2_idx == max_idx or r2_idx == r_idx:
                        continue
                    new_max = max(new_max, route_distance(r2))
                if new_max < route_max():
                    routes[max_idx] = new_route_long
                    routes[r_idx] = new_other_route
                    moved = True
                    report_if_better()
                    break
        if not moved:
            break

    # 2-opt improvement on each route
    for i in range(truck_count):
        route = routes[i]
        if len(route) < 3:
            continue
        for _ in range(len(route)):
            improved = False
            for a in range(len(route)-1):
                for b in range(a+2, len(route)+1):
                    if b - a < 2:
                        continue
                    new_route = route[:a] + route[a:b][::-1] + route[b:]
                    if route_distance(new_route) < route_distance(route):
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                routes[i] = route
                report_if_better()
            else:
                break

    report_if_better()

    full_routes = []
    for r in routes:
        if len(r) == 0:
            full_routes.append([0, 0])
        else:
            full_routes.append([0] + r + [0])
    return full_routes