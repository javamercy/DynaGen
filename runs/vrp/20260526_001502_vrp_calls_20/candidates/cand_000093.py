import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def insert_cost(route, node):
        best_cost = float('inf')
        best_pos = -1
        for i in range(1, len(route)):
            cost = distance_matrix[route[i-1], node] + distance_matrix[node, route[i]] - distance_matrix[route[i-1], route[i]]
            if cost < best_cost:
                best_cost = cost
                best_pos = i
        return best_cost, best_pos

    global_best_max = float('inf')
    global_best_routes = None
    num_restarts = 10

    for restart in range(num_restarts):
        customers = list(range(1, n))
        random.shuffle(customers)

        routes = [[0, 0] for _ in range(truck_count)]
        current_dist = [0.0 for _ in range(truck_count)]
        unassigned = customers[:]

        # Adaptive lambda: initial
        lambda_balance = 0.5

        # Regret-2 with adaptive squared penalty
        while unassigned:
            # Update lambda based on current imbalance
            max_dist = max(current_dist) if current_dist else 0.0
            min_dist = min(current_dist) if current_dist else 0.0
            if max_dist > 0:
                lambda_balance = 0.5 * (1 + (max_dist - min_dist) / (max_dist + 1e-6))
            else:
                lambda_balance = 0.5
            lambda_balance = max(0.3, min(lambda_balance, 1.0))

            best_cust = None
            best_regret = -1.0
            best_route_idx = None
            best_pos = None
            best_cost_val = None
            for cust in unassigned:
                costs_info = []
                for r in range(truck_count):
                    base_cost, pos = insert_cost(routes[r], cust)
                    pen = lambda_balance * (current_dist[r] ** 2)
                    total_cost = base_cost + pen
                    costs_info.append((total_cost, r, pos, base_cost))
                costs_info.sort(key=lambda x: x[0])
                best_cost = costs_info[0][0]
                second_best = costs_info[1][0] if len(costs_info) >= 2 else best_cost
                regret = second_best - best_cost
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = costs_info[0][1]
                    best_pos = costs_info[0][2]
                    best_cost_val = costs_info[0][3]
            routes[best_route_idx].insert(best_pos, best_cust)
            current_dist[best_route_idx] += best_cost_val
            unassigned.remove(best_cust)

        best_max = max(current_dist)
        local_best_routes = [list(r) for r in routes]
        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = [list(r) for r in routes]
            # Assuming report_best_vrp is defined externally; if not, skip
            # report_best_vrp(routes)

        # Post-construction balancing: try to move customer from longest to shortest route
        for _ in range(5 * (n - 1)):  # bounded
            max_route_idx = max(range(truck_count), key=lambda i: current_dist[i])
            min_route_idx = min(range(truck_count), key=lambda i: current_dist[i])
            if max_route_idx == min_route_idx:
                break
            max_route = routes[max_route_idx]
            min_route = routes[min_route_idx]
            if len(max_route) <= 2:
                break
            # find best customer from max_route to move to min_route
            best_gain = 0.0
            best_cust = None
            best_pos_max = -1
            best_pos_min = -1
            for idx in range(1, len(max_route)-1):
                cust = max_route[idx]
                # remove cost from max_route
                remove_cost = distance_matrix[max_route[idx-1], cust] + distance_matrix[cust, max_route[idx+1]] - distance_matrix[max_route[idx-1], max_route[idx+1]]
                # insert into min_route
                ins_cost, ins_pos = insert_cost(min_route, cust)
                new_max_dist = current_dist[max_route_idx] - remove_cost
                new_min_dist = current_dist[min_route_idx] + ins_cost
                other_dists = [current_dist[i] for i in range(truck_count) if i not in (max_route_idx, min_route_idx)]
                new_overall_max = max(new_max_dist, new_min_dist, *other_dists)
                if new_overall_max < best_max:
                    gain = best_max - new_overall_max
                    if gain > best_gain:
                        best_gain = gain
                        best_cust = cust
                        best_pos_max = idx
                        best_pos_min = ins_pos
            if best_cust is not None:
                # apply move
                routes[max_route_idx] = max_route[:best_pos_max] + max_route[best_pos_max+1:]
                routes[min_route_idx] = min_route[:best_pos_min] + [best_cust] + min_route[best_pos_min:]
                current_dist[max_route_idx] -= distance_matrix[max_route[best_pos_max-1], best_cust] + distance_matrix[best_cust, max_route[best_pos_max+1]] - distance_matrix[max_route[best_pos_max-1], max_route[best_pos_max+1]]
                current_dist[min_route_idx] += best_gain + (current_dist[min_route_idx] - ...) # simplified; recalc
                # Recompute distances for changed routes to be safe
                current_dist[max_route_idx] = route_distance(routes[max_route_idx])
                current_dist[min_route_idx] = route_distance(routes[min_route_idx])
                best_max = max(current_dist)
                local_best_routes = [list(r) for r in routes]
                if best_max < global_best_max:
                    global_best_max = best_max
                    global_best_routes = [list(r) for r in routes]
                    # report_best_vrp(routes)
            else:
                break

        # Local search: relocate and swap
        max_iters = 5 * (n - 1) * truck_count
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

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
                        cost, pos = insert_cost(route2, cust)
                        new_dist2 = current_dist[r2] + cost
                        other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        if new_max < best_max:
                            routes[r1] = new_route1
                            routes[r2] = route2[:pos] + [cust] + route2[pos:]
                            current_dist[r1] = new_dist1
                            current_dist[r2] = new_dist2
                            best_max = new_max
                            local_best_routes = [list(r) for r in routes]
                            if new_max < global_best_max:
                                global_best_max = new_max
                                global_best_routes = [list(r) for r in routes]
                                # report_best_vrp(routes)
                            improved = True
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
                            other_dists = [current_dist[i] for i in range(truck_count) if i not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                current_dist[r1] = new_dist1
                                current_dist[r2] = new_dist2
                                best_max = new_max
                                local_best_routes = [list(r) for r in routes]
                                if new_max < global_best_max:
                                    global_best_max = new_max
                                    global_best_routes = [list(r) for r in routes]
                                    # report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        if best_max < global_best_max:
            global_best_max = best_max
            global_best_routes = local_best_routes

    # Ensure final routes have exactly truck_count routes
    while len(global_best_routes) < truck_count:
        global_best_routes.append([0, 0])
    return global_best_routes