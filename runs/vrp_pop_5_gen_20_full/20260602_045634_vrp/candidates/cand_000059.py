import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]

    def route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    global_best_routes = None
    global_best_max = float('inf')

    # Regret-based construction
    routes = [[0,0] for _ in range(truck_count)]
    unassigned = list(range(1, n))

    while unassigned:
        best_cust = None
        best_regret = -1e9
        best_route_idx = None
        best_pos = None
        best_max_val = float('inf')
        for cust in unassigned:
            best_max_for_cust = float('inf')
            second_best_max = float('inf')
            best_r = None
            best_p = None
            for r_idx in range(truck_count):
                route = routes[r_idx]
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    new_dist = route_distance(new_route)
                    other_max = max(route_distance(r) for i,r in enumerate(routes) if i != r_idx)
                    new_max = max(new_dist, other_max)
                    if new_max < best_max_for_cust:
                        second_best_max = best_max_for_cust
                        best_max_for_cust = new_max
                        best_r = r_idx
                        best_p = pos
                    elif new_max < second_best_max:
                        second_best_max = new_max
            regret = second_best_max - best_max_for_cust if second_best_max != float('inf') else float('inf')
            # Tie-breaking: higher regret, then larger best_max_for_cust, then smaller cust
            if (regret > best_regret or 
                (regret == best_regret and best_max_for_cust > best_max_val) or
                (regret == best_regret and best_max_for_cust == best_max_val and (best_cust is None or cust < best_cust))):
                best_regret = regret
                best_max_val = best_max_for_cust
                best_cust = cust
                best_route_idx = best_r
                best_pos = best_p

        routes[best_route_idx].insert(best_pos, best_cust)
        unassigned.remove(best_cust)

    best_routes = [r[:] for r in routes]
    best_max = max(route_distance(r) for r in routes)
    report_best_vrp(best_routes)

    # VNS local search focusing on max route
    max_iter = (n-1) * truck_count * 5
    no_improve_count = 0
    for _ in range(max_iter):
        improved = False
        # Identify route with maximum distance
        max_dist = 0
        max_route_idx = 0
        for i, r in enumerate(routes):
            d = route_distance(r)
            if d > max_dist:
                max_dist = d
                max_route_idx = i
        route_max = routes[max_route_idx]

        # Relocate customer from max route to another
        if len(route_max) > 2:
            for pos in range(1, len(route_max)-1):
                cust = route_max[pos]
                new_max_route = route_max[:pos] + route_max[pos+1:]
                new_max_dist = route_distance(new_max_route)
                for other_idx in range(truck_count):
                    if other_idx == max_route_idx:
                        continue
                    other_route = routes[other_idx]
                    for other_pos in range(1, len(other_route)):
                        new_other = other_route[:other_pos] + [cust] + other_route[other_pos:]
                        new_other_dist = route_distance(new_other)
                        other_max = max(route_distance(routes[i]) for i in range(truck_count) if i not in (max_route_idx, other_idx))
                        new_overall_max = max(new_max_dist, new_other_dist, other_max)
                        if new_overall_max < best_max:
                            routes[max_route_idx] = new_max_route
                            routes[other_idx] = new_other
                            best_max = new_overall_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        if improved:
            no_improve_count = 0
            continue

        # Swap customers between max route and another
        if not improved:
            for other_idx in range(truck_count):
                if other_idx == max_route_idx:
                    continue
                other_route = routes[other_idx]
                if len(route_max) > 2 and len(other_route) > 2:
                    for i in range(1, len(route_max)-1):
                        for j in range(1, len(other_route)-1):
                            cust_i = route_max[i]
                            cust_j = other_route[j]
                            new_max_route = route_max[:i] + [cust_j] + route_max[i+1:]
                            new_other_route = other_route[:j] + [cust_i] + other_route[j+1:]
                            d1 = route_distance(new_max_route)
                            d2 = route_distance(new_other_route)
                            other_max = max(route_distance(routes[k]) for k in range(truck_count) if k not in (max_route_idx, other_idx))
                            new_overall_max = max(d1, d2, other_max)
                            if new_overall_max < best_max:
                                routes[max_route_idx] = new_max_route
                                routes[other_idx] = new_other_route
                                best_max = new_overall_max
                                best_routes = [r[:] for r in routes]
                                report_best_vrp(best_routes)
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break

        if improved:
            no_improve_count = 0
            continue

        # 2-opt on max route
        if not improved:
            if len(route_max) > 3:
                for i in range(1, len(route_max)-2):
                    for j in range(i+1, len(route_max)-1):
                        new_route = route_max[:i] + route_max[i:j+1][::-1] + route_max[j+1:]
                        new_dist = route_distance(new_route)
                        other_max = max(route_distance(routes[k]) for k in range(truck_count) if k != max_route_idx)
                        new_overall_max = max(new_dist, other_max)
                        if new_overall_max < best_max:
                            routes[max_route_idx] = new_route
                            best_max = new_overall_max
                            best_routes = [r[:] for r in routes]
                            report_best_vrp(best_routes)
                            improved = True
                            break
                    if improved:
                        break

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= 5:
            break

    if best_max < global_best_max:
        global_best_max = best_max
        global_best_routes = [r[:] for r in best_routes]
        report_best_vrp(global_best_routes)

    if global_best_routes is None:
        global_best_routes = [r[:] for r in best_routes]
        global_best_max = best_max
        report_best_vrp(global_best_routes)

    return global_best_routes