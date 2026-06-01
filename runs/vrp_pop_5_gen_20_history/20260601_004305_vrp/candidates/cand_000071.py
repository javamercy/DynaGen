import numpy as np
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        report_best_vrp(routes)
        return routes

    # --- Construction: regret-2 insertion ---
    def route_dist(route):
        return sum(distance_matrix[route[k], route[k+1]] for k in range(len(route)-1))

    def compute_max(routes):
        return max(route_dist(r) for r in routes)

    # initialize empty routes
    routes = [[0,0] for _ in range(truck_count)]
    unassigned = list(range(1, n))
    while unassigned:
        best_cust = None
        best_regret = -1
        best_move = None  # (customer, route_idx, pos, resulting_routes)
        for c in unassigned:
            best_val = math.inf
            second_val = math.inf
            best_idx = -1
            best_pos = -1
            for ri, route in enumerate(routes):
                for pos in range(1, len(route)):
                    # compute new max if insert c at pos in route ri
                    new_route = route[:pos] + [c] + route[pos:]
                    new_dist = route_dist(new_route)
                    other_dists = [route_dist(r) for i2, r in enumerate(routes) if i2 != ri]
                    cand_max = max(new_dist, *other_dists)
                    if cand_max < best_val:
                        second_val = best_val
                        best_val = cand_max
                        best_idx = ri
                        best_pos = pos
                    elif cand_max < second_val:
                        second_val = cand_max
            if best_val == math.inf:
                continue
            regret = second_val - best_val
            if regret > best_regret or (regret == best_regret and c < best_cust):
                best_regret = regret
                best_cust = c
                best_move = (c, best_idx, best_pos)
        if best_cust is None:
            # fallback: just insert into any route
            best_cust = unassigned[0]
            best_idx = 0
            best_pos = 1
        c, ri, pos = best_move
        routes[ri].insert(pos, c)
        unassigned.remove(c)
    # ensure no empty routes (truck_count <= m), but some may be [0,0] if less customers?
    # actually all routes have at least one customer because we inserted all
    current_routes = [list(r) for r in routes]
    current_max = compute_max(current_routes)
    best_routes = [list(r) for r in current_routes]
    best_max = current_max
    report_best_vrp(best_routes)

    # --- Tabu Search ---
    # tabu list: dict key -> expiration iteration
    tabu = {}
    tabu_tenure = 10
    iteration = 0
    max_iter = 1000  # bounded by instance size
    no_improve = 0
    while iteration < max_iter and no_improve < 100:
        iteration += 1
        # generate neighborhood: relocate and swap
        best_move = None
        best_new_max = math.inf
        # relocate moves
        for src_ri, src_route in enumerate(current_routes):
            if len(src_route) <= 2:
                continue
            for pos_i in range(1, len(src_route)-1):
                cust = src_route[pos_i]
                for dst_ri in range(truck_count):
                    if dst_ri == src_ri:
                        continue
                    dst_route = current_routes[dst_ri]
                    for pos_j in range(1, len(dst_route)):
                        new_src = src_route[:pos_i] + src_route[pos_i+1:]
                        new_dst = dst_route[:pos_j] + [cust] + dst_route[pos_j:]
                        new_max = max(route_dist(new_src), route_dist(new_dst))
                        for ri2, r in enumerate(current_routes):
                            if ri2 not in (src_ri, dst_ri):
                                new_max = max(new_max, route_dist(r))
                        # check tabu
                        attr = (cust, src_ri)
                        tabu_active = attr in tabu and tabu[attr] >= iteration
                        if tabu_active and new_max >= best_max:
                            continue
                        if new_max < best_new_max or (new_max == best_new_max and (cust < best_move[0] or (cust == best_move[0] and src_ri < best_move[1]) or (cust == best_move[0] and src_ri == best_move[1] and dst_ri < best_move[2]) or (cust == best_move[0] and src_ri == best_move[1] and dst_ri == best_move[2] and pos_j < best_move[3]))):
                            best_new_max = new_max
                            best_move = (cust, src_ri, dst_ri, pos_i, pos_j, 'relocate')
        # swap moves
        for ri1, route1 in enumerate(current_routes):
            if len(route1) <= 2:
                continue
            for pos_i in range(1, len(route1)-1):
                cust1 = route1[pos_i]
                for ri2 in range(ri1+1, truck_count):
                    route2 = current_routes[ri2]
                    if len(route2) <= 2:
                        continue
                    for pos_j in range(1, len(route2)-1):
                        cust2 = route2[pos_j]
                        new_route1 = route1[:pos_i] + [cust2] + route1[pos_i+1:]
                        new_route2 = route2[:pos_j] + [cust1] + route2[pos_j+1:]
                        new_max = max(route_dist(new_route1), route_dist(new_route2))
                        for ri3, r in enumerate(current_routes):
                            if ri3 not in (ri1, ri2):
                                new_max = max(new_max, route_dist(r))
                        # check tabu for both customers
                        attr1 = (cust1, ri1)
                        attr2 = (cust2, ri2)
                        tabu1 = attr1 in tabu and tabu[attr1] >= iteration
                        tabu2 = attr2 in tabu and tabu[attr2] >= iteration
                        if (tabu1 or tabu2) and new_max >= best_max:
                            continue
                        # tie-breaking: use cust1, then ri1, then cust2, then ri2, then pos_i, pos_j
                        if new_max < best_new_max or (new_max == best_new_max and (cust1 < best_move[0] or (cust1 == best_move[0] and ri1 < best_move[1]) or (cust1 == best_move[0] and ri1 == best_move[1] and cust2 < best_move[2]) or (cust1 == best_move[0] and ri1 == best_move[1] and cust2 == best_move[2] and ri2 < best_move[3]) or (cust1 == best_move[0] and ri1 == best_move[1] and cust2 == best_move[2] and ri2 == best_move[3] and pos_i < best_move[4]) or (cust1 == best_move[0] and ri1 == best_move[1] and cust2 == best_move[2] and ri2 == best_move[3] and pos_i == best_move[4] and pos_j < best_move[5]))):
                            best_new_max = new_max
                            best_move = (cust1, ri1, ri2, cust2, pos_i, pos_j, 'swap')
        if best_move is None:
            break
        # apply best move
        if best_move[6] == 'relocate':
            cust, src_ri, dst_ri, pos_i, pos_j, _ = best_move
            src_route = current_routes[src_ri]
            new_src = src_route[:pos_i] + src_route[pos_i+1:]
            dst_route = current_routes[dst_ri]
            new_dst = dst_route[:pos_j] + [cust] + dst_route[pos_j:]
            current_routes[src_ri] = new_src
            current_routes[dst_ri] = new_dst
            # add to tabu
            tabu[(cust, src_ri)] = iteration + tabu_tenure
        else:  # swap
            cust1, ri1, ri2, cust2, pos_i, pos_j, _ = best_move
            route1 = current_routes[ri1]
            route2 = current_routes[ri2]
            new_route1 = route1[:pos_i] + [cust2] + route1[pos_i+1:]
            new_route2 = route2[:pos_j] + [cust1] + route2[pos_j+1:]
            current_routes[ri1] = new_route1
            current_routes[ri2] = new_route2
            tabu[(cust1, ri1)] = iteration + tabu_tenure
            tabu[(cust2, ri2)] = iteration + tabu_tenure
        # update best
        if best_new_max < best_max:
            best_max = best_new_max
            best_routes = [list(r) for r in current_routes]
            report_best_vrp(best_routes)
            no_improve = 0
        else:
            no_improve += 1
        # clean expired tabu
        to_delete = [k for k, v in tabu.items() if v < iteration]
        for k in to_delete:
            del tabu[k]

    return best_routes