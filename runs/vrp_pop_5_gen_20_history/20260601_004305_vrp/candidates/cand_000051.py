import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    customers = set(range(1, n))
    k = truck_count
    if k >= n - 1:
        routes = [[0, i, 0] for i in customers]
        while len(routes) < k:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # Helper functions
    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i]][route[i+1]]
        return d

    def compute_max(routes):
        return max(route_dist(r) for r in routes)

    def copy_routes(routes):
        return [list(r) for r in routes]

    # Initial solution: greedy insertion (random customer order)
    cust_list = list(range(1, n))
    random.shuffle(cust_list)
    routes = [[0, 0] for _ in range(k)]
    for c in cust_list:
        best_max = math.inf
        best_ri = -1
        best_pos = -1
        for ri in range(k):
            route = routes[ri]
            for pos in range(1, len(route)):
                new_route = route[:pos] + [c] + route[pos:]
                new_dist = route_dist(new_route)
                other_dists = [route_dist(routes[j]) for j in range(k) if j != ri]
                cand_max = max(new_dist, *other_dists)
                if cand_max < best_max or (cand_max == best_max and (ri < best_ri or (ri == best_ri and pos < best_pos))):
                    best_max = cand_max
                    best_ri = ri
                    best_pos = pos
        routes[best_ri].insert(best_pos, c)

    best_routes = copy_routes(routes)
    best_max = compute_max(routes)
    report_best_vrp(best_routes)

    # Tabu search parameters
    max_iter = 2000
    tenure = 10
    tabu = {}  # (move_signature) -> remaining iterations
    current_routes = copy_routes(routes)
    current_max = compute_max(current_routes)

    for it in range(max_iter):
        improved = False

        # Phase 1: Relocate moves (move a customer from one route to another)
        best_move = None
        best_new_max = math.inf
        for ri_src in range(k):
            src_route = current_routes[ri_src]
            if len(src_route) <= 2:
                continue
            for pos_i in range(1, len(src_route)-1):
                cust = src_route[pos_i]
                for ri_dst in range(k):
                    if ri_dst == ri_src:
                        continue
                    dst_route = current_routes[ri_dst]
                    for pos_j in range(1, len(dst_route)):
                        new_src = src_route[:pos_i] + src_route[pos_i+1:]
                        new_dst = dst_route[:pos_j] + [cust] + dst_route[pos_j:]
                        new_routes = [current_routes[j] for j in range(k)]
                        new_routes[ri_src] = new_src
                        new_routes[ri_dst] = new_dst
                        new_max = compute_max(new_routes)
                        # Tabu check: is moving cust from ri_src to ri_dst tabu?
                        move_sig = (cust, ri_src)  # forbids moving cust back to ri_src
                        tabu_remaining = tabu.get(move_sig, 0)
                        aspiration = new_max < best_max
                        if (tabu_remaining == 0 or aspiration):
                            if new_max < best_new_max or (new_max == best_new_max and (cust < best_move[0] if best_move else True)):
                                best_new_max = new_max
                                best_move = (cust, ri_src, ri_dst, pos_i, pos_j, new_routes)
        if best_move:
            cust, ri_src, ri_dst, pos_i, pos_j, new_routes = best_move
            current_routes = new_routes
            current_max = best_new_max
            # Update tabu: forbid moving this cust back to ri_src
            tabu[(cust, ri_src)] = tenure + 1
            if current_max < best_max:
                best_max = current_max
                best_routes = copy_routes(current_routes)
                report_best_vrp(best_routes)
            improved = True

        # Phase 2: Swap moves (exchange two customers from different routes)
        if not improved:  # skip if already improved in this iteration? Optional, but to be systematic we do all phases anyway
            best_move = None
            best_new_max = math.inf
            for ri1 in range(k):
                route1 = current_routes[ri1]
                if len(route1) <= 2:
                    continue
                for pos1 in range(1, len(route1)-1):
                    cust1 = route1[pos1]
                    for ri2 in range(ri1+1, k):
                        route2 = current_routes[ri2]
                        if len(route2) <= 2:
                            continue
                        for pos2 in range(1, len(route2)-1):
                            cust2 = route2[pos2]
                            new_route1 = route1[:pos1] + [cust2] + route1[pos1+1:]
                            new_route2 = route2[:pos2] + [cust1] + route2[pos2+1:]
                            new_routes = [current_routes[j] for j in range(k)]
                            new_routes[ri1] = new_route1
                            new_routes[ri2] = new_route2
                            new_max = compute_max(new_routes)
                            # Tabu check: swap moves: forbid moving cust1 from ri1 to ri2 and vice versa? For simplicity, we forbid the two customers from returning to original route.
                            move_sig1 = (cust1, ri1)
                            move_sig2 = (cust2, ri2)
                            tabu1 = tabu.get(move_sig1, 0)
                            tabu2 = tabu.get(move_sig2, 0)
                            aspiration = new_max < best_max
                            if (tabu1 == 0 and tabu2 == 0) or aspiration:
                                if new_max < best_new_max or (new_max == best_new_max and (cust1 < best_move[0] if best_move else True)):
                                    best_new_max = new_max
                                    best_move = (cust1, cust2, ri1, ri2, pos1, pos2, new_routes)
            if best_move:
                cust1, cust2, ri1, ri2, pos1, pos2, new_routes = best_move
                current_routes = new_routes
                current_max = best_new_max
                tabu[(cust1, ri1)] = tenure + 1
                tabu[(cust2, ri2)] = tenure + 1
                if current_max < best_max:
                    best_max = current_max
                    best_routes = copy_routes(current_routes)
                    report_best_vrp(best_routes)
                improved = True

        # Phase 3: 2-opt moves (within one route)
        if not improved:
            best_move = None
            best_new_max = math.inf
            for ri in range(k):
                route = current_routes[ri]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_routes = [current_routes[t] for t in range(k)]
                        new_routes[ri] = new_route
                        new_max = compute_max(new_routes)
                        # Tabu: forbid the reversed edge? Complex. We'll allow any 2-opt as non-tabu (only apply if improves best or not tabu)
                        # For 2-opt, we set a tabu on the first and last edge of the reversed segment? For simplicity, no tabu for 2-opt.
                        # Not a perfect tabu, but different from parents.
                        if new_max < best_new_max or (new_max == best_new_max and (ri < best_move[0] if best_move else True)):
                            best_new_max = new_max
                            best_move = (ri, i, j, new_route)
            if best_move:
                ri, i, j, new_route = best_move
                current_routes[ri] = new_route
                current_max = best_new_max
                # No tabu update for 2-opt (could add but skip)
                if current_max < best_max:
                    best_max = current_max
                    best_routes = copy_routes(current_routes)
                    report_best_vrp(best_routes)
                improved = True

        # Decrease tabu tenures
        to_delete = []
        for key in tabu:
            tabu[key] -= 1
            if tabu[key] <= 0:
                to_delete.append(key)
        for key in to_delete:
            del tabu[key]

    return best_routes