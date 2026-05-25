import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if n == 1:
        return [[0, 0] for _ in range(truck_count)]
    random.seed(0)

    def route_dist(route):
        if len(route) <= 1:
            return 0.0
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    def best_insert(route, node):
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

    # Construction: regret-2 with squared penalty
    customers = list(range(1, n))
    random.shuffle(customers)
    routes = [[0, 0] for _ in range(truck_count)]
    dists = [0.0 for _ in range(truck_count)]
    unassigned = customers[:]
    lambda_bal = 0.5
    while unassigned:
        best_cust = None
        best_regret = -1.0
        best_route = None
        best_pos = None
        best_cost_val = None
        for cust in unassigned:
            costs = []
            for r in range(truck_count):
                base, pos = best_insert(routes[r], cust)
                pen = lambda_bal * (dists[r] ** 2)
                costs.append((base + pen, r, pos, base))
            costs.sort(key=lambda x: x[0])
            regret = (costs[1][0] if len(costs) > 1 else costs[0][0]) - costs[0][0]
            if regret > best_regret:
                best_regret = regret
                best_cust = cust
                best_route = costs[0][1]
                best_pos = costs[0][2]
                best_cost_val = costs[0][3]
        routes[best_route].insert(best_pos, best_cust)
        dists[best_route] += best_cost_val
        unassigned.remove(best_cust)
        avg = sum(dists) / truck_count
        maxd = max(dists)
        imbalance = maxd - avg
        lambda_bal = min(1.0, max(0.1, imbalance / max(avg, 1e-9)))

    best_max = max(dists)
    local_best_routes = [list(r) for r in routes]
    if best_max < global_best_max:
        global_best_max = best_max
        global_best_routes = [list(r) for r in routes]
        report_best_vrp(routes)

    # Perturbation and local search cycles
    for cycle in range(5):
        route_lengths = [len(r) for r in routes]
        if min(route_lengths) < 3:
            break
        max_idx = max(range(truck_count), key=lambda i: dists[i])
        remove_count = max(1, int(0.3 * (len(routes[max_idx]) - 2)))
        remove_count = min(remove_count, len(routes[max_idx]) - 2)
        if remove_count <= 0:
            break
        candidates = list(range(1, len(routes[max_idx])-1))
        random.shuffle(candidates)
        removed = [routes[max_idx][i] for i in candidates[:remove_count]]
        routes[max_idx] = [0] + [node for node in routes[max_idx][1:-1] if node not in removed] + [0]
        for r in range(truck_count):
            dists[r] = route_dist(routes[r])

        # Repair using regret-2
        unassigned = removed[:]
        random.shuffle(unassigned)
        lambda_bal = 0.5
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_route = None
            best_pos = None
            best_cost_val = None
            for cust in unassigned:
                costs = []
                for r in range(truck_count):
                    base, pos = best_insert(routes[r], cust)
                    pen = lambda_bal * (dists[r] ** 2)
                    costs.append((base + pen, r, pos, base))
                costs.sort(key=lambda x: x[0])
                regret = (costs[1][0] if len(costs) > 1 else costs[0][0]) - costs[0][0]
                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_route = costs[0][1]
                    best_pos = costs[0][2]
                    best_cost_val = costs[0][3]
            routes[best_route].insert(best_pos, best_cust)
            dists[best_route] += best_cost_val
            unassigned.remove(best_cust)
            avg = sum(dists) / truck_count
            maxd = max(dists)
            imbalance = maxd - avg
            lambda_bal = min(1.0, max(0.1, imbalance / max(avg, 1e-9)))

        # Local search
        max_iters = 10 * (n - 1) * truck_count
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
                    new_dist1 = route_dist(new_route1)
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        cost, pos = best_insert(route2, cust)
                        new_dist2 = dists[r2] + cost
                        other = [dists[k] for k in range(truck_count) if k not in (r1, r2)]
                        new_max = max(new_dist1, new_dist2, *other)
                        if new_max < global_best_max:
                            routes[r1] = new_route1
                            routes[r2] = route2[:pos] + [cust] + route2[pos:]
                            dists[r1] = new_dist1
                            dists[r2] = new_dist2
                            global_best_max = new_max
                            global_best_routes = [list(r) for r in routes]
                            report_best_vrp(routes)
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
                            new_dist1 = route_dist(new_route1)
                            new_dist2 = route_dist(new_route2)
                            other = [dists[k] for k in range(truck_count) if k not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other)
                            if new_max < global_best_max:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                dists[r1] = new_dist1
                                dists[r2] = new_dist2
                                global_best_max = new_max
                                global_best_routes = [list(r) for r in routes]
                                report_best_vrp(routes)
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
            # Intra-2opt
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                best_imp = 0
                best_i = best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_dist(new_route)
                        if new_dist < dists[r] - 1e-9:
                            imp = dists[r] - new_dist
                            if imp > best_imp:
                                best_imp = imp
                                best_i, best_j = i, j
                if best_imp > 0:
                    new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                    routes[r] = new_route
                    dists[r] = route_dist(new_route)
                    new_max = max(dists)
                    if new_max < global_best_max:
                        global_best_max = new_max
                        global_best_routes = [list(r) for r in routes]
                        report_best_vrp(routes)
                    improved = True
                    break
            if improved:
                continue
            # Cross-2opt
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 2:
                        continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i+1] + route2[j+1:]
                            new2 = route2[:j+1] + route1[i+1:]
                            new_dist1 = route_dist(new1)
                            new_dist2 = route_dist(new2)
                            other = [dists[k] for k in range(truck_count) if k not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other)
                            if new_max < global_best_max:
                                routes[r1] = new1
                                routes[r2] = new2
                                dists[r1] = new_dist1
                                dists[r2] = new_dist2
                                global_best_max = new_max
                                global_best_routes = [list(r) for r in routes]
                                report_best_vrp(routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        current_max = max(dists)
        if current_max < best_max:
            best_max = current_max
            local_best_routes = [list(r) for r in routes]

    return global_best_routes if global_best_routes is not None else local_best_routes