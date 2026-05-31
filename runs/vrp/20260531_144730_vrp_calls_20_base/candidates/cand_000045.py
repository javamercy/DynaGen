import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    n_cust = len(customers)
    if truck_count >= n_cust:
        routes = [[0, 0] for _ in range(truck_count)]
        for i, cust in enumerate(customers):
            routes[i] = [0, cust, 0]
        return routes

    # 1. Greedy insertion initial solution: build routes one by one
    routes = [[0, 0] for _ in range(truck_count)]
    unvisited = set(customers)
    # Helper to compute max distance of a set of routes
    def max_dist(route_list):
        maxd = 0.0
        for r in route_list:
            d = 0.0
            for a, b in zip(r[:-1], r[1:]):
                d += distance_matrix[a, b]
            if d > maxd:
                maxd = d
        return maxd

    # For each customer, insert in best position across all routes
    for cust in customers:
        best_max = float('inf')
        best_route_idx = -1
        best_pos = -1
        for ri, route in enumerate(routes):
            for pos in range(1, len(route)):
                new_route = route[:pos] + [cust] + route[pos:]
                temp_routes = routes[:ri] + [new_route] + routes[ri+1:]
                cur_max = max_dist(temp_routes)
                if cur_max < best_max or (cur_max == best_max and cust < customers[0]):  # tie-break by customer index? Actually we need deterministic: first encountered route index then position
                    # Use tuple (cur_max, ri, pos) for ordering; but to keep simple, just update on strictly less
                    best_max = cur_max
                    best_route_idx = ri
                    best_pos = pos
        # insert at best position
        routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]

    # Ensure all routes start/end with 0 (they already do)
    # Fill empty routes if needed (already have truck_count routes)

    best_routes = [r[:] for r in routes]
    report_best_vrp(best_routes)

    # Helper functions (same as parent)
    def route_dist(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for a, b in zip(route[:-1], route[1:]):
            d += distance_matrix[a, b]
        return d

    def two_opt(route):
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_dist(new_route) < route_dist(best):
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best

    # 2. Ruin and recreate improvement
    max_iter = n_cust * 5
    for _ in range(max_iter):
        old_max = max_dist(routes)
        # Select customers with highest contribution
        contribs = []
        for ridx, route in enumerate(routes):
            if len(route) <= 2:
                continue
            for pos in range(1, len(route)-1):
                cust = route[pos]
                c = distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos+1]]
                contribs.append((-c, cust, ridx, pos))
        contribs.sort()
        ruin_size = max(1, n_cust // 10)
        to_ruin = [t[1] for t in contribs[:ruin_size]]

        saved_routes = [r[:] for r in routes]

        # Remove customers from routes
        for cust in to_ruin:
            for ridx, route in enumerate(routes):
                if cust in route:
                    new_route = [x for x in route if x != cust]
                    if len(new_route) < 2 or new_route[0] != 0:
                        new_route = [0] + new_route
                    if new_route[-1] != 0:
                        new_route = new_route + [0]
                    if len(new_route) == 2 and new_route[0] == 0 and new_route[1] == 0:
                        pass
                    elif len(new_route) == 2:
                        new_route = [0, 0]
                    routes[ridx] = new_route
                    break

        # Regret-2 insertion
        unrouted = set(to_ruin)
        current_routes = [r[:] for r in routes]
        while unrouted:
            best_cust = None
            best_regret = -1.0
            best_insert = (None, None)
            for cust in sorted(unrouted):  # deterministic order
                costs = []
                for ri, route in enumerate(current_routes):
                    best_cost = float('inf')
                    best_pos = -1
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        temp_routes = current_routes[:]
                        temp_routes[ri] = new_route
                        new_max = max_dist(temp_routes)
                        if new_max < best_cost:
                            best_cost = new_max
                            best_pos = pos
                    costs.append((best_cost, best_pos, ri))
                costs.sort(key=lambda x: (x[0], x[2]))  # tie-breaking by route index
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = 0.0
                if regret > best_regret + 1e-12:
                    best_regret = regret
                    best_cust = cust
                    best_insert = (costs[0][2], costs[0][1])
            ri, pos = best_insert
            route = current_routes[ri]
            if len(route) == 2 and route[0] == 0 and route[1] == 0:
                current_routes[ri] = [0, best_cust, 0]
            else:
                current_routes[ri] = route[:pos] + [best_cust] + route[pos:]
            unrouted.remove(best_cust)

        new_max = max_dist(current_routes)
        if new_max < old_max - 1e-9:
            routes = current_routes
            for idx in range(truck_count):
                routes[idx] = two_opt(routes[idx])
            if new_max < max_dist(best_routes) - 1e-9:
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
        else:
            routes = saved_routes

    # 3. Relocate and exchange improvement
    max_iter2 = n_cust * 2
    for _ in range(max_iter2):
        moved = False
        # Relocate
        for from_route in range(truck_count):
            for to_route in range(truck_count):
                if from_route == to_route or len(routes[from_route]) <= 2:
                    continue
                for cust_pos in range(1, len(routes[from_route])-1):
                    for insert_pos in range(1, len(routes[to_route])):
                        new_routes = [r[:] for r in routes]
                        cust = new_routes[from_route].pop(cust_pos)
                        new_routes[to_route].insert(insert_pos, cust)
                        new_max = max_dist(new_routes)
                        if new_max < max_dist(best_routes) - 1e-9:
                            routes = new_routes
                            for idx in range(truck_count):
                                routes[idx] = two_opt(routes[idx])
                            if max_dist(routes) < max_dist(best_routes) - 1e-9:
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    break
            if moved:
                break
        if moved:
            continue
        # Exchange
        for r1, r2 in combinations(range(truck_count), 2):
            if len(routes[r1]) <= 2 or len(routes[r2]) <= 2:
                continue
            for p1 in range(1, len(routes[r1])-1):
                for p2 in range(1, len(routes[r2])-1):
                    new_routes = [r[:] for r in routes]
                    new_routes[r1][p1], new_routes[r2][p2] = new_routes[r2][p2], new_routes[r1][p1]
                    new_max = max_dist(new_routes)
                    if new_max < max_dist(best_routes) - 1e-9:
                        routes = new_routes
                        for idx in range(truck_count):
                            routes[idx] = two_opt(routes[idx])
                        if max_dist(routes) < max_dist(best_routes) - 1e-9:
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break
        if not moved:
            break

    return best_routes