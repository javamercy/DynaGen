import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    if truck_count >= n - 1:
        routes = [[0, c, 0] for c in range(1, n)]
        routes += [[0, 0]] * (truck_count - len(routes))
        return routes

    # helper functions
    def route_dist(route):
        d = 0.0
        for i in range(len(route) - 1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def compute_max_route_dist(routes, dists):
        return max(dists) if dists else 0.0

    def deep_copy_routes(routes):
        return [list(r) for r in routes]

    # regret insertion function: takes a set of unassigned customers and modifies routes and route_dists in place
    def regret_insert(unassigned, routes, route_dists):
        while unassigned:
            best_cust = None
            best_regret = -1.0
            best_route_idx = -1
            best_pos = -1
            for cust in unassigned:
                best_inc = float('inf')
                second_inc = float('inf')
                best_r = -1
                best_p = -1
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        prev = route[pos-1]
                        nxt = route[pos]
                        inc = distance_matrix[prev][cust] + distance_matrix[cust][nxt] - distance_matrix[prev][nxt]
                        new_route_dist = route_dists[r_idx] + inc
                        other_max = max(route_dists[:r_idx] + route_dists[r_idx+1:]) if truck_count > 1 else 0.0
                        new_max = max(new_route_dist, other_max)
                        if new_max < best_inc - 1e-12:
                            second_inc = best_inc
                            best_inc = new_max
                            best_r = r_idx
                            best_p = pos
                        elif new_max < second_inc - 1e-12:
                            second_inc = new_max
                regret = second_inc - best_inc if second_inc < float('inf') else best_inc
                if regret > best_regret or (abs(regret - best_regret) < 1e-12 and (best_cust is None or cust < best_cust)):
                    best_regret = regret
                    best_cust = cust
                    best_route_idx = best_r
                    best_pos = best_p
            # insert best_cust
            route = routes[best_route_idx]
            prev = route[best_pos-1]
            nxt = route[best_pos]
            inc = distance_matrix[prev][best_cust] + distance_matrix[best_cust][nxt] - distance_matrix[prev][nxt]
            route_dists[best_route_idx] += inc
            route.insert(best_pos, best_cust)
            unassigned.remove(best_cust)

    # local search: inter-route relocate and intra-route 2-opt
    def local_search(routes, route_dists):
        improved = True
        while improved:
            improved = False
            # inter-route relocate (move one customer to another route)
            for src_idx in range(truck_count):
                src_route = routes[src_idx]
                if len(src_route) <= 2:
                    continue
                # consider each customer in src_route (excluding depots)
                for cust_pos in range(1, len(src_route)-1):
                    cust = src_route[cust_pos]
                    # try moving to other routes
                    for dst_idx in range(truck_count):
                        if dst_idx == src_idx:
                            continue
                        dst_route = routes[dst_idx]
                        best_pos = -1
                        best_new_max = route_dists[src_idx]  # dummy to compare
                        # evaluate each insertion position
                        for pos in range(1, len(dst_route)):
                            # cost change if we remove cust from src and insert at pos in dst
                            prev_src = src_route[cust_pos-1]
                            next_src = src_route[cust_pos+1]
                            delta_src = -distance_matrix[prev_src][cust] - distance_matrix[cust][next_src] + distance_matrix[prev_src][next_src]
                            prev_dst = dst_route[pos-1]
                            next_dst = dst_route[pos]
                            delta_dst = distance_matrix[prev_dst][cust] + distance_matrix[cust][next_dst] - distance_matrix[prev_dst][next_dst]
                            new_src_dist = route_dists[src_idx] + delta_src
                            new_dst_dist = route_dists[dst_idx] + delta_dst
                            other_max = max(route_dists[:src_idx] + route_dists[src_idx+1:dst_idx] + route_dists[dst_idx+1:]) if truck_count > 2 else 0.0
                            new_max = max(new_src_dist, new_dst_dist, other_max)
                            if new_max < best_new_max - 1e-12:
                                best_new_max = new_max
                                best_pos = pos
                        if best_pos != -1:
                            # perform move
                            old_src_dist = route_dists[src_idx]
                            old_dst_dist = route_dists[dst_idx]
                            # remove cust from src
                            src_route.pop(cust_pos)
                            route_dists[src_idx] += delta_src
                            # insert cust into dst at best_pos
                            dst_route.insert(best_pos, cust)
                            route_dists[dst_idx] += delta_dst
                            improved = True
                            break  # restart while loop after change
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # intra-route 2-opt for each route
            for idx in range(truck_count):
                route = routes[idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        # reverse segment from i to j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_dist(new_route)
                        if new_dist < route_dists[idx] - 1e-12:
                            # check new max
                            other_max = max(route_dists[:idx] + route_dists[idx+1:]) if truck_count > 1 else 0.0
                            new_max = max(new_dist, other_max)
                            if new_max < compute_max_route_dist(routes, route_dists) - 1e-12:
                                route_dists[idx] = new_dist
                                routes[idx] = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    # initial construction: regret insertion
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = set(range(1, n))
    regret_insert(unassigned, routes, route_dists)
    best_routes = deep_copy_routes(routes)
    best_max = compute_max_route_dist(routes, route_dists)
    report_best_vrp(best_routes)

    # main optimization loop
    max_iter = min(n * 10, 500)
    initial_threshold = 0.1 * best_max
    if initial_threshold < 1e-9:
        initial_threshold = 1.0
    threshold = initial_threshold
    no_improve_iter = 0
    restart_interval = int(0.2 * max_iter)
    if restart_interval < 1:
        restart_interval = 1
    rng = random.Random(0)  # deterministic random

    for iteration in range(max_iter):
        # destroy the longest route
        current_max = max(route_dists)
        max_routes = [i for i, d in enumerate(route_dists) if abs(d - current_max) < 1e-12]
        destroy_route_idx = min(max_routes)
        destroy_route = routes[destroy_route_idx]
        if len(destroy_route) <= 2:
            continue
        removed_customers = destroy_route[1:-1]
        # save current state
        old_routes = deep_copy_routes(routes)
        old_dists = list(route_dists)
        # recreate: remove customers from that route and reinsert
        routes[destroy_route_idx] = [0, 0]
        route_dists[destroy_route_idx] = 0.0
        # reinsert sorted (deterministic)
        removed_sorted = sorted(removed_customers)
        regret_insert(set(removed_sorted), routes, route_dists)
        # apply local search
        local_search(routes, route_dists)
        new_max = compute_max_route_dist(routes, route_dists)
        # acceptance
        if new_max <= best_max + threshold:
            if new_max < best_max - 1e-12:
                best_routes = deep_copy_routes(routes)
                best_max = new_max
                report_best_vrp(best_routes)
                no_improve_iter = 0
            else:
                no_improve_iter += 1
        else:
            # revert
            routes = old_routes
            route_dists = old_dists
            no_improve_iter += 1
        # update threshold
        threshold = initial_threshold * (1 - (iteration+1)/max_iter)
        if threshold < 0:
            threshold = 0.0
        # restart if stagnant
        if no_improve_iter >= restart_interval:
            # perturb best solution: move 10% of customers to different routes
            candidates = list(range(1, n))
            rng.shuffle(candidates)
            num_perturb = max(1, int(0.1 * (n-1)))
            perturb_custs = candidates[:num_perturb]
            # remove them from their current routes
            new_routes = deep_copy_routes(best_routes)
            new_dists = [route_dist(r) for r in new_routes]
            removed = []
            for cust in perturb_custs:
                for idx, route in enumerate(new_routes):
                    if cust in route:
                        pos = route.index(cust)
                        # remove customer
                        prev = route[pos-1]
                        nxt = route[pos+1]
                        delta = -distance_matrix[prev][cust] - distance_matrix[cust][nxt] + distance_matrix[prev][nxt]
                        new_dists[idx] += delta
                        route.pop(pos)
                        removed.append(cust)
                        break
            # reinsert shifted customers using regret insertion
            regret_insert(set(removed), new_routes, new_dists)
            # set current to perturbed
            routes = new_routes
            route_dists = new_dists
            # reset threshold
            threshold = initial_threshold
            no_improve_iter = 0
            # optionally report this perturbation? Not needed

    return best_routes