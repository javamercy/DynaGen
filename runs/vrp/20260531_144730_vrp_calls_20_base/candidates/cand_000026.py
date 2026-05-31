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

    def route_dist(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for a, b in zip(route[:-1], route[1:]):
            d += distance_matrix[a, b]
        return d

    def max_dist(routes_list):
        return max(route_dist(r) for r in routes_list)

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
                    if route_dist(new_route) < route_dist(best) - 1e-12:
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best

    # Step 1: Initial solution via regret-2 construction (modified from parent2)
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = set(customers)

    def best_insertion(c, routes_list, route_dists_list):
        # returns (best_new_max, best_route_idx, best_pos, second_best_new_max)
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes_list):
            if len(route) < 2:
                continue
            other_max = 0.0
            for j, d in enumerate(route_dists_list):
                if j != r_idx and d > other_max:
                    other_max = d
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists_list[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    while unassigned:
        candidates = []
        for c in sorted(unassigned):
            best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
            candidates.append((-regret, c, best_route, best_pos, best_new_max))
        candidates.sort(key=lambda x: (x[0], x[1]))  # deterministic tie by customer index
        _, c, best_route, best_pos, _ = candidates[0]
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)
        unassigned.remove(c)
        report_best_vrp(routes)

    # Step 2: Intra-route 2-opt
    for idx in range(truck_count):
        routes[idx] = two_opt(routes[idx])
        route_dists[idx] = route_dist(routes[idx])
    report_best_vrp(routes)

    best_routes = [r[:] for r in routes]

    # Step 3: Ruin-recreate (adapted from parent1, uses regret-2 insertion)
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
        # Remove customers
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

        # Regret-2 insertion (like parent1's ruin-recreate but using max distance)
        unrouted = set(to_ruin)
        current_routes = [r[:] for r in routes]
        while unrouted:
            best_cust = None
            best_regret = -1.0
            best_insert = (None, None)
            for cust in sorted(unrouted):
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
                costs.sort(key=lambda x: (x[0], x[2]))
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

    # Step 4: Relocate and exchange improvement (from parent1)
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