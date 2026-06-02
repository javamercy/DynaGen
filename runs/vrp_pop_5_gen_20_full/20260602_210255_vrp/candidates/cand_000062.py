import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))

    orders = [
        customers,
        customers[::-1],
        sorted(customers, key=lambda c: distance_matrix[0][c]),
        sorted(customers, key=lambda c: -distance_matrix[0][c])
    ]

    best_routes = None
    best_max = float('inf')

    for order in orders:
        # Construction
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0] * truck_count
        unvisited = set(range(1, n))
        order_pos = {c: i for i, c in enumerate(order)}

        while unvisited:
            best_cust = None
            best_regret = -1
            best_cost = float('inf')
            best_route_idx = None
            best_pos = None
            for cust in unvisited:
                costs = []
                for r_idx, route in enumerate(routes):
                    if route_dists[r_idx] == 0 and len(route) == 2:
                        inc = distance_matrix[0][cust] + distance_matrix[cust][0]
                        new_max = max(inc, max(d for i, d in enumerate(route_dists) if i != r_idx))
                        costs.append((new_max, r_idx, 1))
                    else:
                        for pos in range(1, len(route)):
                            prev = route[pos-1]
                            nxt = route[pos]
                            inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                            new_dist = route_dists[r_idx] + inc
                            new_max = max(new_dist, max(d for i, d in enumerate(route_dists) if i != r_idx))
                            costs.append((new_max, r_idx, pos))
                if not costs:
                    continue
                costs.sort(key=lambda x: x[0])
                best = costs[0][0]
                second = costs[1][0] if len(costs) > 1 else best
                regret = second - best
                if regret > best_regret + 1e-9:
                    best_regret = regret
                    best_cost = best
                    best_cust = cust
                    best_route_idx = costs[0][1]
                    best_pos = costs[0][2]
                elif abs(regret - best_regret) < 1e-9:
                    if best < best_cost - 1e-9:
                        best_cost = best
                        best_cust = cust
                        best_route_idx = costs[0][1]
                        best_pos = costs[0][2]
                    elif abs(best - best_cost) < 1e-9:
                        if order_pos[cust] < order_pos.get(best_cust, float('inf')):
                            best_cust = cust
                            best_route_idx = costs[0][1]
                            best_pos = costs[0][2]
                            best_cost = best
            if best_cust is None:
                break
            route = routes[best_route_idx]
            route.insert(best_pos, best_cust)
            route_dists[best_route_idx] = sum(
                distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
            unvisited.remove(best_cust)

        report_best_vrp(routes)

        # Local search
        improved = True
        iter_count = 0
        max_iter = n * 5
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            best_move = None
            best_new_max = max(route_dists)

            # Relocate moves
            for r_idx, route in enumerate(routes):
                if len(route) <= 2:
                    continue
                for pos in range(1, len(route) - 1):
                    cust = route[pos]
                    new_route_r = route[:pos] + route[pos+1:]
                    dist_r_new = sum(
                        distance_matrix[new_route_r[i], new_route_r[i+1]] for i in range(len(new_route_r)-1))
                    for r2_idx, route2 in enumerate(routes):
                        if r2_idx == r_idx:
                            continue
                        for pos2 in range(1, len(route2)):
                            inc = distance_matrix[route2[pos2-1]][cust] + distance_matrix[cust][route2[pos2]] - distance_matrix[route2[pos2-1]][route2[pos2]]
                            new_route_r2 = route2[:pos2] + [cust] + route2[pos2:]
                            dist_r2_new = route_dists[r2_idx] + inc
                            new_max = max(
                                dist_r_new,
                                dist_r2_new,
                                max(route_dists[i] for i in range(len(routes)) if i not in (r_idx, r2_idx))
                            )
                            if new_max < best_new_max - 1e-9:
                                best_new_max = new_max
                                best_move = ('relocate', r_idx, pos, r2_idx, pos2,
                                             new_route_r, new_route_r2, dist_r_new, dist_r2_new)

            # 2-opt* moves
            for r1 in range(len(routes)):
                for r2 in range(r1 + 1, len(routes)):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(len(route1) - 1):
                        for j in range(len(route2) - 1):
                            new1 = route1[:i+1] + route2[j+1:]
                            new2 = route2[:j+1] + route1[i+1:]
                            dist1_new = sum(
                                distance_matrix[new1[k], new1[k+1]] for k in range(len(new1)-1))
                            dist2_new = sum(
                                distance_matrix[new2[k], new2[k+1]] for k in range(len(new2)-1))
                            new_max = max(
                                dist1_new,
                                dist2_new,
                                max(route_dists[k] for k in range(len(routes)) if k not in (r1, r2))
                            )
                            if new_max < best_new_max - 1e-9:
                                best_new_max = new_max
                                best_move = ('2opt', r1, r2, i, j, new1, new2, dist1_new, dist2_new)

            if best_move is not None:
                if best_move[0] == 'relocate':
                    _, r_idx, pos, r2_idx, pos2, new_r, new_r2, d1, d2 = best_move
                    routes[r_idx] = new_r
                    routes[r2_idx] = new_r2
                    route_dists[r_idx] = d1
                    route_dists[r2_idx] = d2
                else:
                    _, r1, r2, i, j, new1, new2, d1, d2 = best_move
                    routes[r1] = new1
                    routes[r2] = new2
                    route_dists[r1] = d1
                    route_dists[r2] = d2
                improved = True
                report_best_vrp(routes)

        final_max = max(route_dists)
        if final_max < best_max - 1e-9:
            best_max = final_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)

    return best_routes