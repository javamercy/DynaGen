import numpy as np
from itertools import combinations

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    routes = [[0, 0] for _ in range(truck_count)]
    route_dist = [0.0 for _ in range(truck_count)]

    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    for r in range(truck_count):
        route_dist[r] = compute_route_distance(routes[r])

    unassigned = set(range(1, n))

    def best_max_and_second(customer):
        best_val = float('inf')
        best_pos = None
        second_val = float('inf')
        for r_idx, route in enumerate(routes):
            curr_dist = route_dist[r_idx]
            for i in range(1, len(route)):
                new_dist = curr_dist - distance_matrix[route[i-1], route[i]] + distance_matrix[route[i-1], customer] + distance_matrix[customer, route[i]]
                other_max = max(route_dist[:r_idx] + route_dist[r_idx+1:], default=0.0)
                cand_max = max(new_dist, other_max)
                if cand_max < best_val:
                    second_val = best_val
                    best_val = cand_max
                    best_pos = (r_idx, i)
                elif cand_max < second_val and cand_max != best_val:
                    second_val = cand_max
        return best_val, second_val, best_pos

    while unassigned:
        best_regret = -1.0
        best_customer = None
        best_insertion = None
        best_best_val = float('inf')
        for c in unassigned:
            best_val, second_val, best_pos = best_max_and_second(c)
            regret = second_val - best_val if second_val != float('inf') else 0.0
            if regret > best_regret or (regret == best_regret and (best_val < best_best_val or (abs(best_val - best_best_val) < 1e-9 and (best_customer is None or c < best_customer)))):
                best_regret = regret
                best_customer = c
                best_insertion = best_pos
                best_best_val = best_val
        r_idx, i = best_insertion
        route = routes[r_idx]
        route.insert(i, best_customer)
        route_dist[r_idx] = compute_route_distance(route)
        unassigned.remove(best_customer)

    report_best_vrp(routes)

    def vnd():
        nonlocal routes, route_dist
        max_iter = n * truck_count
        vnd_iter = 0
        improved = True
        while improved and vnd_iter < max_iter:
            improved = False
            current_max = max(route_dist)
            # intra-route 2-opt
            for r_idx in range(truck_count):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if len(new_route) != len(route):
                            continue
                        if new_dist < route_dist[r_idx]:
                            new_max = max(new_dist, max(route_dist[:r_idx] + route_dist[r_idx+1:], default=0.0))
                            if new_max < current_max - 1e-9:
                                routes[r_idx] = new_route
                                route_dist[r_idx] = new_dist
                                current_max = new_max
                                improved = True
                                report_best_vrp(routes)
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                vnd_iter += 1
                continue
            # inter-route relocate
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2:
                    continue
                for i in range(1, len(route1)-1):
                    cust = route1[i]
                    for r2 in range(truck_count):
                        if r2 == r1:
                            continue
                        route2 = routes[r2]
                        for pos in range(1, len(route2)):
                            prev_rem = route1[i-1]
                            next_rem = route1[i+1]
                            removed_cost = distance_matrix[prev_rem, cust] + distance_matrix[cust, next_rem] - distance_matrix[prev_rem, next_rem]
                            new_dist_r1 = route_dist[r1] - removed_cost
                            prev_ins = route2[pos-1]
                            next_ins = route2[pos]
                            added_cost = distance_matrix[prev_ins, cust] + distance_matrix[cust, next_ins] - distance_matrix[prev_ins, next_ins]
                            new_dist_r2 = route_dist[r2] + added_cost
                            other_max = max(route_dist[:r1] + route_dist[r1+1:r2] + route_dist[r2+1:], default=0.0)
                            new_max = max(new_dist_r1, new_dist_r2, other_max)
                            if new_max < current_max - 1e-9:
                                new_route1 = route1[:i] + route1[i+1:]
                                if len(new_route1) == 2:
                                    new_route1 = [0, 0]
                                new_route2 = route2[:pos] + [cust] + route2[pos:]
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                route_dist[r1] = compute_route_distance(new_route1)
                                route_dist[r2] = compute_route_distance(new_route2)
                                current_max = new_max
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                vnd_iter += 1
                continue
            # inter-route swap
            for r1 in range(truck_count):
                route1 = routes[r1]
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            cust1 = route1[i]
                            cust2 = route2[j]
                            new_route1 = route1[:i] + [cust2] + route1[i+1:]
                            new_route2 = route2[:j] + [cust1] + route2[j+1:]
                            new_dist1 = compute_route_distance(new_route1)
                            new_dist2 = compute_route_distance(new_route2)
                            other_max = max(route_dist[:r1] + route_dist[r1+1:r2] + route_dist[r2+1:], default=0.0)
                            new_max = max(new_dist1, new_dist2, other_max)
                            if new_max < current_max - 1e-9:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                route_dist[r1] = new_dist1
                                route_dist[r2] = new_dist2
                                current_max = new_max
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                vnd_iter += 1
                continue
            # inter-route 2-opt* (cross)
            for r1 in range(truck_count):
                route1 = routes[r1]
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i] + route2[j:]
                            new2 = route2[:j] + route1[i:]
                            new_dist1 = compute_route_distance(new1)
                            new_dist2 = compute_route_distance(new2)
                            other_max = max(route_dist[:r1] + route_dist[r1+1:r2] + route_dist[r2+1:], default=0.0)
                            new_max = max(new_dist1, new_dist2, other_max)
                            if new_max < current_max - 1e-9:
                                routes[r1] = new1
                                routes[r2] = new2
                                route_dist[r1] = new_dist1
                                route_dist[r2] = new_dist2
                                current_max = new_max
                                improved = True
                                report_best_vrp(routes)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            vnd_iter += 1

    # initial VND
    vnd()

    # deterministic restart: remove customer with largest contribution to max route distance from each route, reinsert using regret-2, reapply VND
    restart_iterations = max(1, min(5, n // (5 * truck_count)))
    for _ in range(restart_iterations):
        removed_customers = []
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            # find customer whose removal yields lowest new max
            best_new_max = max(route_dist)
            best_cust = None
            best_pos = None
            for i in range(1, len(route)-1):
                cust = route[i]
                # compute new distance for this route after removal
                new_route = route[:i] + route[i+1:]
                if len(new_route) == 2:
                    new_dist = 0.0
                else:
                    new_dist = compute_route_distance(new_route)
                other_max = max(route_dist[:r_idx] + route_dist[r_idx+1:], default=0.0)
                new_max = max(new_dist, other_max)
                if new_max < best_new_max - 1e-9:
                    best_new_max = new_max
                    best_cust = cust
                    best_pos = i
            if best_cust is not None:
                removed_customers.append((r_idx, best_pos, best_cust))
        if not removed_customers:
            continue
        # remove them (process in reverse order to avoid index shift)
        removed_customers.sort(key=lambda x: (x[0], -x[1]))
        for r_idx, pos, cust in removed_customers:
            route = routes[r_idx]
            if route[pos] == cust:
                route.pop(pos)
                route_dist[r_idx] = compute_route_distance(route)
        # reinsert using regret-2 with tie-breaking by best insertion value
        unassigned = set(c for _, _, c in removed_customers)
        while unassigned:
            best_regret = -1.0
            best_customer = None
            best_insertion = None
            best_best_val = float('inf')
            for c in unassigned:
                best_val, second_val, best_pos = best_max_and_second(c)
                regret = second_val - best_val if second_val != float('inf') else 0.0
                if regret > best_regret or (regret == best_regret and (best_val < best_best_val or (abs(best_val - best_best_val) < 1e-9 and (best_customer is None or c < best_customer)))):
                    best_regret = regret
                    best_customer = c
                    best_insertion = best_pos
                    best_best_val = best_val
            if best_customer is None:
                break
            r_idx, i = best_insertion
            route = routes[r_idx]
            route.insert(i, best_customer)
            route_dist[r_idx] = compute_route_distance(route)
            unassigned.remove(best_customer)
        report_best_vrp(routes)
        vnd()

    return routes