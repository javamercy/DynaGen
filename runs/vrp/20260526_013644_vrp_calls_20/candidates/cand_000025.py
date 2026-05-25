import numpy as np
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    depot = 0
    customers = list(range(1, n))
    unassigned = set(customers)
    routes = [[depot, depot] for _ in range(truck_count)]
    
    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    current_dists = [0.0 for _ in range(truck_count)]
    
    def insertion_cost_and_pos(cust, route_idx):
        route = routes[route_idx]
        best_new_max = float('inf')
        best_pos = -1
        base_dist = current_dists[route_idx]
        for pos in range(1, len(route)):
            i = route[pos-1]
            j = route[pos]
            new_dist = base_dist - distance_matrix[i, j] + distance_matrix[i, cust] + distance_matrix[cust, j]
            new_max = max(current_dists[:route_idx] + [new_dist] + current_dists[route_idx+1:])
            if new_max < best_new_max:
                best_new_max = new_max
                best_pos = pos
        if best_pos == -1:
            return float('inf'), -1
        return best_new_max, best_pos
    
    # Construction
    while unassigned:
        best_regret = -1.0
        best_cust = -1
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')
        
        for cust in list(unassigned):
            costs = []
            for r_idx in range(truck_count):
                cost, pos = insertion_cost_and_pos(cust, r_idx)
                if pos != -1:
                    costs.append((cost, r_idx, pos))
            if not costs:
                continue
            costs.sort(key=lambda x: x[0])
            if len(costs) == 1:
                regret = 1e9
            else:
                regret = costs[1][0] - costs[0][0]
            best_cost_here = costs[0][0]
            if (regret > best_regret or
                (regret == best_regret and best_cost_here < best_cost) or
                (regret == best_regret and best_cost_here == best_cost and cust < best_cust)):
                best_regret = regret
                best_cust = cust
                best_cost = best_cost_here
                best_route_idx = costs[0][1]
                best_pos = costs[0][2]
        
        # Insert
        route = routes[best_route_idx]
        route.insert(best_pos, best_cust)
        # Update current_dists
        current_dists[best_route_idx] = route_dist(route)
        unassigned.remove(best_cust)
    
    best_routes = [list(r) for r in routes]
    best_max = max(current_dists)
    report_best_vrp(best_routes)
    
    n_customers = n - 1
    max_search_iters = 3 * n_customers
    random.seed(42)
    perturbation_count = 0
    max_perturbations = 3
    
    for iteration in range(max_search_iters):
        improved = False
        
        # Inter-route relocate (best improvement)
        best_move = None
        best_new_max = best_max
        for r_idx in range(truck_count):
            route = routes[r_idx]
            if len(route) <= 3:
                continue
            for cust in route[1:-1]:
                new_route = [x for x in route if x != cust]
                new_dist_after_remove = route_dist(new_route)
                temp_dists = current_dists[:]
                temp_dists[r_idx] = new_dist_after_remove
                for other_idx in range(truck_count):
                    if other_idx == r_idx:
                        continue
                    other_route = routes[other_idx]
                    cost, pos = insertion_cost_and_pos(cust, other_idx)
                    # Actually we need to compute with temp_dists? Better compute directly
                    # Recalculate with temp_dists
                    for pos2 in range(1, len(other_route)):
                        i = other_route[pos2-1]
                        j = other_route[pos2]
                        new_other_dist = current_dists[other_idx] - distance_matrix[i, j] + distance_matrix[i, cust] + distance_matrix[cust, j]
                        new_max = max(temp_dists[:other_idx] + [new_other_dist] + temp_dists[other_idx+1:])
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = (r_idx, other_idx, cust, pos2, new_route, new_other_dist, new_dist_after_remove)
        if best_move is not None:
            r_idx, other_idx, cust, pos, new_route, new_other_dist, new_dist_remove = best_move
            routes[r_idx] = new_route
            current_dists[r_idx] = new_dist_remove
            other_route = routes[other_idx]
            other_route.insert(pos, cust)
            current_dists[other_idx] = new_other_dist
            best_max = best_new_max
            improved = True
            report_best_vrp(routes)
        else:
            # Inter-route swap (best improvement)
            best_swap = None
            best_new_max = best_max
            for r1 in range(truck_count):
                route1 = routes[r1]
                if len(route1) <= 3:
                    continue
                for cust1 in route1[1:-1]:
                    for r2 in range(r1+1, truck_count):
                        route2 = routes[r2]
                        if len(route2) <= 3:
                            continue
                        for cust2 in route2[1:-1]:
                            # Remove both
                            new_route1 = [x for x in route1 if x != cust1]
                            new_route2 = [x for x in route2 if x != cust2]
                            # Insert swapped
                            # Insert cust2 into new_route1 best pos
                            best1 = float('inf')
                            best_pos1 = -1
                            for pos in range(1, len(new_route1)):
                                i = new_route1[pos-1]
                                j = new_route1[pos]
                                d = route_dist(new_route1[:pos] + [cust2] + new_route1[pos:])
                                # compute dist quickly
                                new_route1_temp = new_route1[:pos] + [cust2] + new_route1[pos:]
                                d1 = route_dist(new_route1_temp)
                                if d1 < best1:
                                    best1 = d1
                                    best_pos1 = pos
                            # Insert cust1 into new_route2 best pos
                            best2 = float('inf')
                            best_pos2 = -1
                            for pos in range(1, len(new_route2)):
                                new_route2_temp = new_route2[:pos] + [cust1] + new_route2[pos:]
                                d2 = route_dist(new_route2_temp)
                                if d2 < best2:
                                    best2 = d2
                                    best_pos2 = pos
                            if best_pos1 == -1 or best_pos2 == -1:
                                continue
                            new_routes = [list(r) for r in routes]
                            new_routes[r1] = new_route1[:best_pos1] + [cust2] + new_route1[best_pos1:]
                            new_routes[r2] = new_route2[:best_pos2] + [cust1] + new_route2[best_pos2:]
                            dists = [route_dist(r) for r in new_routes]
                            new_max = max(dists)
                            if new_max < best_new_max:
                                best_new_max = new_max
                                best_swap = (r1, r2, cust1, cust2, best_pos1, best_pos2, new_routes)
            if best_swap is not None:
                routes = best_swap[6]
                current_dists = [route_dist(r) for r in routes]
                best_max = best_new_max
                improved = True
                report_best_vrp(routes)
            else:
                # Intra-route 2-opt
                for r_idx in range(truck_count):
                    route = routes[r_idx]
                    if len(route) <= 4:
                        continue
                    best_imp = None
                    best_dist = current_dists[r_idx]
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            new_dist = route_dist(new_route)
                            if new_dist < best_dist:
                                best_dist = new_dist
                                best_imp = (i, j, new_route)
                    if best_imp is not None:
                        routes[r_idx] = best_imp[2]
                        current_dists[r_idx] = best_dist
                        improved = True
                        new_max = max(current_dists)
                        if new_max < best_max:
                            best_max = new_max
                            report_best_vrp(routes)
        
        if not improved:
            # Perturbation
            if perturbation_count >= max_perturbations:
                break
            perturbation_count += 1
            # Perform random relocations
            for _ in range(3):
                # pick random customer
                all_custs = [c for route in routes for c in route[1:-1]]
                if not all_custs:
                    break
                cust = random.choice(all_custs)
                # find its current route
                current_route_idx = None
                for idx, route in enumerate(routes):
                    if cust in route:
                        current_route_idx = idx
                        break
                if current_route_idx is None:
                    continue
                # remove from current route
                route = routes[current_route_idx]
                route.remove(cust)
                current_dists[current_route_idx] = route_dist(route)
                # insert into random route at random position
                target_idx = random.randint(0, truck_count-1)
                target_route = routes[target_idx]
                pos = random.randint(1, len(target_route)-1)
                target_route.insert(pos, cust)
                current_dists[target_idx] = route_dist(target_route)
            # After perturbation, recalc best_max
            best_max = max(current_dists)
            # Do not report perturbation as improvement unless it actually is better
            if best_max < best_max_before:
                report_best_vrp(routes)
            # Update best_max_before for next checks
            # Continue loop to re-apply local search
    
    # Ensure exactly truck_count routes
    result = []
    for r in routes:
        if len(r) <= 2:
            result.append([0, 0])
        else:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
            result.append(r)
    while len(result) < truck_count:
        result.append([0, 0])
    return result