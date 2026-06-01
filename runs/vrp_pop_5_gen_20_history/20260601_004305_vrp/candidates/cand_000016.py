import numpy as np
import math
import random

def solve_vrp(distance_matrix, truck_count):
    n = distance_matrix.shape[0]
    # ---- initial solution via Clarke-Wright savings (like parent 9) ----
    routes = [[0, i, 0] for i in range(1, n)]
    while len(routes) < truck_count:
        routes.append([0, 0])
    
    def route_dist(route):
        d = 0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d
    
    def compute_savings(routes):
        savings = []
        for i, r_i in enumerate(routes):
            if len(r_i) == 2:
                continue
            last_i = r_i[-2]
            first_i = r_i[1]
            for j, r_j in enumerate(routes):
                if i == j or len(r_j) == 2:
                    continue
                first_j = r_j[1]
                last_j = r_j[-2]
                s1 = distance_matrix[0, last_i] + distance_matrix[0, first_j] - distance_matrix[last_i, first_j]
                savings.append((s1, i, j, 0))
                s2 = distance_matrix[0, last_j] + distance_matrix[0, first_i] - distance_matrix[last_j, first_i]
                savings.append((s2, i, j, 1))
        savings.sort(reverse=True, key=lambda x: x[0])
        return savings
    
    def merge_routes(routes, i, j, mtype):
        if mtype == 0:
            new_route = routes[i][:-1] + routes[j][1:]
        else:
            new_route = routes[j][:-1] + routes[i][1:]
        if i > j:
            del routes[i]
            del routes[j]
        else:
            del routes[j]
            del routes[i]
        routes.append(new_route)
        return routes
    
    while len(routes) > truck_count:
        savings = compute_savings(routes)
        if not savings:
            break
        for saving, i, j, mtype in savings:
            if i >= len(routes) or j >= len(routes):
                continue
            routes = merge_routes(routes, i, j, mtype)
            break
    
    best_routes = [list(r) for r in routes]
    best_max = max(route_dist(r) for r in routes)
    report_best_vrp(best_routes)
    
    # ---- VNS main loop ----
    max_iters = n * 2
    no_improve_vns = 0
    neighborhood_idx = 0  # 0: 2-opt, 1: relocate, 2: swap
    source_idx = 0
    route_order = sorted(range(truck_count), key=lambda i: route_dist(routes[i]), reverse=True)
    
    for _ in range(max_iters):
        # Shaking: apply random relocate of up to 3 customers (inter-route)
        shaken = [list(r) for r in routes]
        num_shakes = random.randint(1, min(3, n-1))
        for _ in range(num_shakes):
            # choose random customer from a random source route (avoid empty)
            src_idx = random.randint(0, truck_count-1)
            while len(shaken[src_idx]) <= 2:
                src_idx = random.randint(0, truck_count-1)
            cust_pos = random.randint(1, len(shaken[src_idx])-2)
            cust = shaken[src_idx].pop(cust_pos)
            # insert into random destination route at random position
            dst_idx = random.randint(0, truck_count-1)
            if dst_idx == src_idx:
                # if same, insert back at random position
                pos = random.randint(1, len(shaken[src_idx])-1)
                shaken[src_idx].insert(pos, cust)
            else:
                pos = random.randint(1, len(shaken[dst_idx])-1)
                shaken[dst_idx].insert(pos, cust)
        # Evaluate shaken solution
        shaken_max = max(route_dist(r) for r in shaken)
        if shaken_max < best_max:
            routes = [list(r) for r in shaken]
            best_max = shaken_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
            no_improve_vns = 0
            continue
        # Local search on shaken solution (best-improvement relocate with adaptive source)
        # Use best-improvement relocate from parent 14 but without changing routes permanently until improvement found
        # We'll work on a copy of shaken
        candidate = [list(r) for r in shaken]
        local_improved = False
        local_iters = truck_count * 2
        for _ in range(local_iters):
            cur_max = max(route_dist(r) for r in candidate)
            # adaptive source
            longest_idx = route_order[source_idx]
            longest = candidate[longest_idx]
            if len(longest) <= 2:
                source_idx = (source_idx + 1) % truck_count
                continue
            best_improvement = 0.0
            best_move = None
            for idx_c in range(1, len(longest)-1):
                cust = longest[idx_c]
                new_longest = longest[:idx_c] + longest[idx_c+1:]
                dist_longest = route_dist(new_longest)
                for other_idx in range(truck_count):
                    if other_idx == longest_idx:
                        continue
                    other = candidate[other_idx]
                    for pos in range(1, len(other)):
                        new_other = other[:pos] + [cust] + other[pos:]
                        dist_other = route_dist(new_other)
                        new_max = max(dist_longest, dist_other)
                        for ri in range(truck_count):
                            if ri != longest_idx and ri != other_idx:
                                d = route_dist(candidate[ri])
                                if d > new_max:
                                    new_max = d
                        improvement = cur_max - new_max
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (cust, idx_c, other_idx, pos)
            if best_move and best_improvement > 0:
                cust, idx_c, other_idx, pos = best_move
                candidate[longest_idx].remove(cust)
                candidate[other_idx].insert(pos, cust)
                local_improved = True
                # update route_order
                route_order = sorted(range(truck_count), key=lambda i: route_dist(candidate[i]), reverse=True)
                source_idx = 0
                # compare with best
                cand_max = max(route_dist(r) for r in candidate)
                if cand_max < best_max:
                    routes = [list(r) for r in candidate]
                    best_max = cand_max
                    best_routes = [list(r) for r in routes]
                    report_best_vrp(best_routes)
                break
            else:
                # no improvement from this source, switch source
                source_idx = (source_idx + 1) % truck_count
                # reset after 5 failures
                if source_idx == 0:
                    # after cycling through all sources, break if no improvement
                    if not local_improved:
                        break
        # After local search, if candidate improved, accept it
        if local_improved:
            # candidate already accepted if it improved best, but we also might accept non-improving? In VNS we accept if shaken not improved, but we did local search on shaken. To keep VNS, we only accept if new best found? Actually standard VNS accepts if solution improves after local search. We'll accept if best improved.
            pass
        # Neighborhood change
        if best_max < max(route_dist(r) for r in shaken):
            # improvement found, continue with same neighborhood
            pass
        else:
            neighborhood_idx = (neighborhood_idx + 1) % 3
            no_improve_vns += 1
        if no_improve_vns > truck_count:
            break
    
    return best_routes