import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    random.seed(0)
    
    def route_distance(route):
        if len(route) <= 1:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def best_insert(route, customer):
        best_dist = float('inf')
        best_pos = -1
        for pos in range(1, len(route)):
            new_dist = route_distance(route[:pos] + [customer] + route[pos:])
            if new_dist < best_dist:
                best_dist = new_dist
                best_pos = pos
        return best_pos, best_dist
    
    def regret_construction(order):
        routes = [[0,0] for _ in range(truck_count)]
        dists = [0.0]*truck_count
        customers = list(order)
        while customers:
            best_cust = None
            best_regret = -float('inf')
            best_route = None
            best_pos = None
            best_new_dist = None
            for cust in customers:
                best_primary = float('inf')
                best_secondary = float('inf')
                second_primary = float('inf')
                second_secondary = float('inf')
                best_route_for_cust = None
                best_pos_for_cust = None
                for r in range(truck_count):
                    pos, new_dist = best_insert(routes[r], cust)
                    current_max = max(dists)
                    new_max = max(current_max, new_dist)
                    # primary: new_max, secondary: new_dist
                    if (new_max, new_dist) < (best_primary, best_secondary):
                        second_primary, second_secondary = best_primary, best_secondary
                        best_primary, best_secondary = new_max, new_dist
                        best_route_for_cust = r
                        best_pos_for_cust = pos
                    elif (new_max, new_dist) < (second_primary, second_secondary):
                        second_primary, second_secondary = new_max, new_dist
                if second_primary == float('inf'):
                    regret_primary = 0
                else:
                    regret_primary = second_primary - best_primary
                if regret_primary > best_regret or (regret_primary == best_regret and best_secondary < best_new_dist):
                    best_regret = regret_primary
                    best_cust = cust
                    best_route = best_route_for_cust
                    best_pos = best_pos_for_cust
                    best_new_dist = best_secondary
            # Insert best_cust
            routes[best_route].insert(best_pos, best_cust)
            dists[best_route] = route_distance(routes[best_route])
            customers.remove(best_cust)
        return routes, dists
    
    def local_search(routes, dists):
        n_cust = sum(len(r)-2 for r in routes)
        max_iters = 10 * n_cust * truck_count
        best_routes = [list(r) for r in routes]
        best_max = max(dists)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Relocate
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2: continue
                for idx in range(1, len(route1)-1):
                    cust = route1[idx]
                    new_route1 = route1[:idx] + route1[idx+1:]
                    new_dist1 = route_distance(new_route1)
                    for r2 in range(truck_count):
                        if r2 == r1: continue
                        route2 = routes[r2]
                        old_dist2 = dists[r2]
                        cost, pos = best_insert(route2, cust)
                        new_dist2 = old_dist2 + cost
                        other_dists = [dists[i] for i in range(truck_count) if i not in (r1,r2)]
                        new_max = max(new_dist1, new_dist2, *other_dists)
                        if new_max < best_max - 1e-9:
                            routes[r1] = new_route1
                            routes[r2] = route2[:pos] + [cust] + route2[pos:]
                            dists[r1] = new_dist1
                            dists[r2] = new_dist2
                            best_max = new_max
                            best_routes = [list(r) for r in routes]
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: continue
            # Swap
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2: continue
                for idx1 in range(1, len(route1)-1):
                    cust1 = route1[idx1]
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        if len(route2) <= 2: continue
                        for idx2 in range(1, len(route2)-1):
                            cust2 = route2[idx2]
                            new_route1 = route1[:idx1] + [cust2] + route1[idx1+1:]
                            new_route2 = route2[:idx2] + [cust1] + route2[idx2+1:]
                            new_dist1 = route_distance(new_route1)
                            new_dist2 = route_distance(new_route2)
                            other_dists = [dists[i] for i in range(truck_count) if i not in (r1,r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-9:
                                routes[r1] = new_route1
                                routes[r2] = new_route2
                                dists[r1] = new_dist1
                                dists[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break
            if improved: continue
            # Intra-route 2-opt
            for r in range(truck_count):
                route = routes[r]
                if len(route) <= 3: continue
                best_improve = 0
                best_i = best_j = -1
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_dist = route_distance(new_route)
                        if new_dist < dists[r] - 1e-9:
                            improvement = dists[r] - new_dist
                            if improvement > best_improve:
                                best_improve = improvement
                                best_i, best_j = i, j
                if best_improve > 1e-9:
                    new_route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                    routes[r] = new_route
                    dists[r] = route_distance(new_route)
                    new_max = max(dists)
                    if new_max < best_max - 1e-9:
                        best_max = new_max
                        best_routes = [list(r) for r in routes]
                    improved = True
                    break
            if improved: continue
            # Cross-route 2-opt
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 2: continue
                for r2 in range(r1+1, truck_count):
                    route2 = routes[r2]
                    if len(route2) <= 2: continue
                    for i in range(1, len(route1)-1):
                        for j in range(1, len(route2)-1):
                            new1 = route1[:i+1] + route2[j+1:]
                            new2 = route2[:j+1] + route1[i+1:]
                            new_dist1 = route_distance(new1)
                            new_dist2 = route_distance(new2)
                            other_dists = [dists[k] for k in range(truck_count) if k not in (r1,r2)]
                            new_max = max(new_dist1, new_dist2, *other_dists)
                            if new_max < best_max - 1e-9:
                                routes[r1] = new1
                                routes[r2] = new2
                                dists[r1] = new_dist1
                                dists[r2] = new_dist2
                                best_max = new_max
                                best_routes = [list(r) for r in routes]
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break
        return best_routes, best_max
    
    best_overall_routes = None
    best_overall_max = float('inf')
    num_restarts = 5
    customers_list = list(range(1, n))
    for restart in range(num_restarts):
        random.shuffle(customers_list)
        routes, dists = regret_construction(customers_list)
        # Perturbation: remove 3 random customers and reinsert
        remove_count = min(3, len(customers_list))
        if remove_count > 0:
            to_remove = random.sample(range(1, n), remove_count)
            for cust in to_remove:
                for r in range(truck_count):
                    if cust in routes[r]:
                        routes[r].remove(cust)
                        dists[r] = route_distance(routes[r])
                        break
            remaining = [c for c in customers_list if c not in to_remove]
            random.shuffle(to_remove)
            reinsert_order = to_remove
            for cust in reinsert_order:
                best_route = None
                best_pos = None
                best_new_max = float('inf')
                best_new_dist = None
                current_max = max(dists)
                for r in range(truck_count):
                    pos, new_dist = best_insert(routes[r], cust)
                    new_max = max(current_max, new_dist)
                    if (new_max, new_dist) < (best_new_max, best_new_dist if best_new_dist is not None else float('inf')):
                        best_new_max = new_max
                        best_new_dist = new_dist
                        best_route = r
                        best_pos = pos
                routes[best_route].insert(best_pos, cust)
                dists[best_route] = route_distance(routes[best_route])
        else:
            pass
        # Local search
        routes, max_dist = local_search(routes, dists)
        if max_dist < best_overall_max - 1e-9:
            best_overall_max = max_dist
            best_overall_routes = [list(r) for r in routes]
            # report_best_vrp(best_overall_routes)
    # Ensure all routes start and end at 0, and customers covered
    # (already satisfied by construction)
    return best_overall_routes