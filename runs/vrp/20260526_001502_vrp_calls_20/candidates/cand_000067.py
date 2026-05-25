import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]

    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_new_max(routes, route_dists, cust, r, pos):
        route = routes[r]
        inc = (distance_matrix[route[pos-1], cust] + distance_matrix[cust, route[pos]] - distance_matrix[route[pos-1], route[pos]])
        new_dist = route_dists[r] + inc
        other_max = max(route_dists[i] for i in range(truck_count) if i != r)
        return max(new_dist, other_max)

    def insert_best(routes, route_dists, cust, best_r, best_pos):
        routes[best_r].insert(best_pos, cust)
        route_dists[best_r] = route_distance(routes[best_r])

    def construct_regret2(rng):
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0 for _ in range(truck_count)]
        unassigned = list(range(1, n))
        rng.shuffle(unassigned)
        cust = unassigned.pop(0)
        best_max = float('inf')
        best_r = -1
        best_pos = -1
        for r in range(truck_count):
            route = routes[r]
            for pos in range(1, len(route)):
                new_max = compute_new_max(routes, route_dists, cust, r, pos)
                if new_max < best_max - 1e-9:
                    best_max = new_max
                    best_r = r
                    best_pos = pos
                elif abs(new_max - best_max) < 1e-9 and r < best_r:
                    best_r = r
                    best_pos = pos
        insert_best(routes, route_dists, cust, best_r, best_pos)
        while unassigned:
            best_regret = -1.0
            best_cust = -1
            best_r1 = -1
            best_pos1 = -1
            for cust in unassigned:
                best_max1 = float('inf')
                best_max2 = float('inf')
                best_r_temp = -1
                best_pos_temp = -1
                for r in range(truck_count):
                    route = routes[r]
                    for pos in range(1, len(route)):
                        new_max = compute_new_max(routes, route_dists, cust, r, pos)
                        if new_max < best_max1 - 1e-9:
                            best_max2 = best_max1
                            best_max1 = new_max
                            best_r_temp = r
                            best_pos_temp = pos
                        elif new_max < best_max2 - 1e-9:
                            best_max2 = new_max
                if best_max2 == float('inf'):
                    best_max2 = best_max1
                regret = best_max2 - best_max1
                if regret > best_regret - 1e-9:
                    best_regret = regret
                    best_cust = cust
                    best_r1 = best_r_temp
                    best_pos1 = best_pos_temp
                elif abs(regret - best_regret) < 1e-9 and cust < best_cust:
                    best_cust = cust
                    best_r1 = best_r_temp
                    best_pos1 = best_pos_temp
            if best_cust == -1:
                break
            insert_best(routes, route_dists, best_cust, best_r1, best_pos1)
            unassigned.remove(best_cust)
        return routes, route_dists

    def local_search(routes, route_dists):
        best_routes = [list(r) for r in routes]
        best_max = max(route_dists)
        try:
            report_best_vrp(best_routes)
        except NameError:
            pass
        max_iters = 10 * n * truck_count
        for _ in range(max_iters):
            improved = False
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
                        for pos in range(1, len(route2)):
                            inc = (distance_matrix[route2[pos-1], cust] + distance_matrix[cust, route2[pos]] - distance_matrix[route2[pos-1], route2[pos]])
                            new_dist2 = route_dists[r2] + inc
                            other_dists = [route_dists[i] for i in range(truck_count) if i not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-9:
                                routes[r1] = new_route1
                                route_dists[r1] = new_dist1
                                routes[r2].insert(pos, cust)
                                route_dists[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                                try:
                                    report_best_vrp(best_routes)
                                except NameError:
                                    pass
                                break
                        if improved:
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
                            other_dists = [route_dists[i] for i in range(truck_count) if i not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-9:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                route_dists[r1] = new_dist1
                                route_dists[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                                try:
                                    report_best_vrp(best_routes)
                                except NameError:
                                    pass
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Intra-route 2-opt
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3:
                    continue
                best_improve = 0.0
                best_i = best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < route_dists[r] - 1e-9:
                            improvement = route_dists[r] - new_dist
                            if improvement > best_improve:
                                best_improve = improvement
                                best_i, best_j = i, j
                if best_improve > 0:
                    new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                    routes[r] = new_route
                    route_dists[r] = route_distance(new_route)
                    new_max = max(route_dists)
                    if new_max < best_max - 1e-9:
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                        try:
                            report_best_vrp(best_routes)
                        except NameError:
                            pass
                    improved = True
                    break
            if improved:
                continue
            # Cross-route 2-opt
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
                            new_dist1 = route_distance(new1)
                            new_dist2 = route_distance(new2)
                            other_dists = [route_dists[k] for k in range(truck_count) if k not in (r1, r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-9:
                                routes[r1] = new1
                                routes[r2] = new2
                                route_dists[r1] = new_dist1
                                route_dists[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                                try:
                                    report_best_vrp(best_routes)
                                except NameError:
                                    pass
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return best_routes, best_max

    def perturb(routes, route_dists, rng):
        num_remove = max(2, int(n * 0.15))
        # Identify longest route(s)
        max_dist = max(route_dists)
        longest_routes = [i for i, d in enumerate(route_dists) if d > max_dist - 1e-9]
        # Collect customers from longest routes first, then others
        all_cust = []
        # First add customers from longest routes
        for r in longest_routes:
            route = routes[r]
            for idx in range(1, len(route)-1):
                all_cust.append((r, idx, routes[r][idx]))
        # Then from other routes
        for r in range(truck_count):
            if r in longest_routes:
                continue
            route = routes[r]
            for idx in range(1, len(route)-1):
                all_cust.append((r, idx, routes[r][idx]))
        if len(all_cust) == 0:
            return routes, route_dists
        num_remove = min(num_remove, len(all_cust))
        # Sort by route index to remove in reverse order later (not critical)
        to_remove = all_cust[:num_remove]
        # Remove in reverse order of index to avoid shifting issues
        to_remove_sorted = sorted(to_remove, key=lambda x: (x[0], x[1]), reverse=True)
        removed_custs = []
        for r, idx, cust in to_remove_sorted:
            routes[r].pop(idx)
            removed_custs.append(cust)
        # Recompute distances
        for r in range(truck_count):
            route_dists[r] = route_distance(routes[r])
        rng.shuffle(removed_custs)
        # Reinsert using best-min-max with tie-breaking favoring balance (smaller route distance before insertion?)
        for cust in removed_custs:
            best_max = float('inf')
            best_r = -1
            best_pos = -1
            for r in range(truck_count):
                route = routes[r]
                for pos in range(1, len(route)):
                    new_max_val = compute_new_max(routes, route_dists, cust, r, pos)
                    if new_max_val < best_max - 1e-9:
                        best_max = new_max_val
                        best_r = r
                        best_pos = pos
                    elif abs(new_max_val - best_max) < 1e-9:
                        # Tie-breaking: prefer route with smaller current distance (balance)
                        if route_dists[r] < route_dists[best_r] - 1e-9:
                            best_r = r
                            best_pos = pos
                        elif abs(route_dists[r] - route_dists[best_r]) < 1e-9 and r < best_r:
                            best_r = r
                            best_pos = pos
            insert_best(routes, route_dists, cust, best_r, best_pos)
        return routes, route_dists

    num_restarts = 5
    cycles = 3
    best_routes_overall = []
    best_max_overall = float('inf')
    for restart_idx in range(num_restarts):
        rng = random.Random(restart_idx * 12345 + 42)
        routes, route_dists = construct_regret2(rng)
        routes, max_dist = local_search(routes, route_dists)
        if max_dist < best_max_overall - 1e-9:
            best_max_overall = max_dist
            best_routes_overall = [list(r) for r in routes]
            try:
                report_best_vrp(best_routes_overall)
            except NameError:
                pass
        for cycle in range(cycles):
            routes_pert, route_dists_pert = perturb(routes, route_dists, rng)
            routes_pert, max_dist_pert = local_search(routes_pert, route_dists_pert)
            if max_dist_pert < best_max_overall - 1e-9:
                best_max_overall = max_dist_pert
                best_routes_overall = [list(r) for r in routes_pert]
                try:
                    report_best_vrp(best_routes_overall)
                except NameError:
                    pass
            routes = routes_pert
            route_dists = [route_distance(r) for r in routes]
    for r in range(truck_count):
        if len(best_routes_overall[r]) == 0:
            best_routes_overall[r] = [0, 0]
    return best_routes_overall