import numpy as np
import random
import math

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    random.seed(0)

    def compute_route_distance(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def evaluate_insertion(route, route_dist, customer, position):
        # position is index in route where customer is inserted (before route[position])
        new_dist = route_dist - distance_matrix[route[position-1], route[position]] \
                   + distance_matrix[route[position-1], customer] \
                   + distance_matrix[customer, route[position]]
        return new_dist

    def best_insertion(customer, routes, route_dists):
        best_val = float('inf')
        best_pos = None
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            current_dist = route_dists[r_idx]
            for i in range(1, len(route)):
                new_dist = evaluate_insertion(route, current_dist, customer, i)
                other_max = max([route_dists[j] for j in range(truck_count) if j != r_idx] or [0.0])
                cand_max = max(new_dist, other_max)
                if cand_max < best_val - 1e-12:
                    best_val = cand_max
                    best_pos = (r_idx, i)
        return best_val, best_pos

    def regret2_construction():
        routes = [[0, 0] for _ in range(truck_count)]
        route_dists = [0.0 for _ in range(truck_count)]
        unassigned = list(range(1, n))
        random.shuffle(unassigned)
        # initial assignment: assign each customer to the best route
        for c in unassigned:
            best_val, best_pos = best_insertion(c, routes, route_dists)
            if best_pos is None:
                # fallback: insert in first route
                routes[0].insert(1, c)
                route_dists[0] = compute_route_distance(routes[0])
            else:
                r_idx, i = best_pos
                routes[r_idx].insert(i, c)
                route_dists[r_idx] = compute_route_distance(routes[r_idx])
        # then iterative regret-2 improvement
        # This is not pure regret, but simpler: just insertion order
        return routes, route_dists

    def insertion_heuristic(removed, routes, route_dists):
        unassigned = list(removed)
        while unassigned:
            best_cust = None
            best_val = float('inf')
            best_pos = None
            for c in unassigned:
                val, pos = best_insertion(c, routes, route_dists)
                if val < best_val - 1e-12:
                    best_val = val
                    best_cust = c
                    best_pos = pos
            if best_pos is None:
                # insert in first route at end
                routes[0].insert(len(routes[0])-1, best_cust)
                route_dists[0] = compute_route_distance(routes[0])
            else:
                r_idx, i = best_pos
                routes[r_idx].insert(i, best_cust)
                route_dists[r_idx] = compute_route_distance(routes[r_idx])
            unassigned.remove(best_cust)

    def local_search(routes, route_dists):
        current_routes = [list(r) for r in routes]
        current_dists = list(route_dists)
        current_max = max(current_dists)
        improved = True
        max_iters = n * n  # finite bound
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            # Intra-route 2-opt
            for r_idx in range(truck_count):
                route = current_routes[r_idx]
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = compute_route_distance(new_route)
                        if new_dist < current_dists[r_idx] - 1e-12:
                            current_dists[r_idx] = new_dist
                            current_routes[r_idx] = new_route
                            new_max = max(current_dists)
                            if new_max < current_max - 1e-12:
                                current_max = new_max
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-relocate
            for r1 in range(truck_count):
                for r2 in range(truck_count):
                    if r1 == r2:
                        continue
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        customer = route1[i]
                        new_route1 = route1[:i] + route1[i+1:]
                        new_dist1 = compute_route_distance(new_route1)
                        best_new_dist2 = float('inf')
                        best_new_route2 = None
                        for j in range(1, len(route2)):
                            new_route2 = route2[:j] + [customer] + route2[j:]
                            d2 = compute_route_distance(new_route2)
                            if d2 < best_new_dist2:
                                best_new_dist2 = d2
                                best_new_route2 = new_route2
                        other_max = max([current_dists[k] for k in range(truck_count) if k not in (r1, r2)] or [0])
                        cand_max = max(new_dist1, best_new_dist2, other_max)
                        if cand_max < current_max - 1e-12:
                            current_routes[r1] = new_route1
                            current_routes[r2] = best_new_route2
                            current_dists[r1] = new_dist1
                            current_dists[r2] = best_new_dist2
                            current_max = cand_max
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Inter-swap
            for r1 in range(truck_count):
                for r2 in range(r1+1, truck_count):
                    route1 = current_routes[r1]
                    route2 = current_routes[r2]
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new_route1 = route1[:i] + [route2[j]] + route1[i+1:]
                            new_route2 = route2[:j] + [route1[i]] + route2[j+1:]
                            new_dist1 = compute_route_distance(new_route1)
                            new_dist2 = compute_route_distance(new_route2)
                            other_max = max([current_dists[k] for k in range(truck_count) if k not in (r1, r2)] or [0])
                            cand_max = max(new_dist1, new_dist2, other_max)
                            if cand_max < current_max - 1e-12:
                                current_routes[r1] = new_route1
                                current_routes[r2] = new_route2
                                current_dists[r1] = new_dist1
                                current_dists[r2] = new_dist2
                                current_max = cand_max
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return current_routes, current_dists, current_max

    def ruin_recreate(routes, route_dists):
        new_routes = [list(r) for r in routes]
        new_dists = list(route_dists)
        # remove from longest route(s)
        sorted_indices = sorted(range(truck_count), key=lambda i: new_dists[i], reverse=True)
        num_routes = min(3, truck_count)
        removed = []
        for idx in sorted_indices[:num_routes]:
            route = new_routes[idx]
            if len(route) <= 2:
                continue
            # remove 30% of customers from that route
            num_remove = max(1, int(0.3 * (len(route) - 2)))
            removable = list(range(1, len(route)-1))
            random.shuffle(removable)
            to_remove = removable[:num_remove]
            to_remove.sort(reverse=True)
            for pos in to_remove:
                removed.append(route.pop(pos))
            new_dists[idx] = compute_route_distance(route)
        if not removed:
            return new_routes, new_dists
        # reinsert using greedy best insertion
        insertion_heuristic(removed, new_routes, new_dists)
        return new_routes, new_dists

    best_routes = None
    best_max_val = float('inf')
    max_restarts = 5

    # initial construction
    routes, route_dists = regret2_construction()
    routes, route_dists, current_max = local_search(routes, route_dists)
    if current_max < best_max_val - 1e-12:
        best_max_val = current_max
        best_routes = routes
        report_best_vrp(best_routes)

    for restart in range(max_restarts - 1):
        # perturb
        routes, route_dists = ruin_recreate(best_routes, [compute_route_distance(r) for r in best_routes])
        routes, route_dists, current_max = local_search(routes, route_dists)
        if current_max < best_max_val - 1e-12:
            best_max_val = current_max
            best_routes = routes
            report_best_vrp(best_routes)

    return best_routes