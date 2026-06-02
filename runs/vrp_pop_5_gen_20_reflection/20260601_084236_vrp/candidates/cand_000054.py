import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    if truck_count >= n:
        routes = [[0, i, 0] for i in range(1, n)]
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    def route_distance(route):
        if len(route) < 2:
            return 0.0
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def max_distance(routes):
        return max(route_distance(r) for r in routes)

    def two_opt(route):
        if len(route) <= 3:
            return route
        improved = True
        best = route[:]
        best_d = route_distance(route)
        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    if j - i == 1:
                        continue
                    new = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d = route_distance(new)
                    if d < best_d - 1e-12:
                        best_d = d
                        best = new
                        improved = True
            route = best
        return best

    def construct_initial(seed_idx):
        random.seed(seed_idx)
        seeds = random.sample(customers, min(truck_count, len(customers)))
        clusters = [[] for _ in range(truck_count)]
        for i, s in enumerate(seeds):
            clusters[i].append(s)
        remaining = [c for c in customers if c not in seeds]
        for cust in remaining:
            best_dist = float('inf')
            best_cluster = 0
            for i, seed in enumerate(seeds):
                d = distance_matrix[cust, seed]
                if d < best_dist - 1e-12:
                    best_dist = d
                    best_cluster = i
            clusters[best_cluster].append(cust)
        routes = []
        for cluster in clusters:
            if not cluster:
                routes.append([0, 0])
            else:
                unvisited = set(cluster)
                current = 0
                tour = [0]
                while unvisited:
                    next_cust = min(unvisited, key=lambda c: distance_matrix[current, c])
                    tour.append(next_cust)
                    unvisited.remove(next_cust)
                    current = next_cust
                tour.append(0)
                routes.append(two_opt(tour))
        return routes

    best_routes = None
    best_max = float('inf')
    max_restarts = max(truck_count, 10)
    for restart in range(max_restarts):
        routes = construct_initial(restart)
        current_max = max_distance(routes)
        if current_max < best_max - 1e-12:
            best_max = current_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        # Intra-route 2-opt one pass
        for t in range(truck_count):
            routes[t] = two_opt(routes[t])
        cur_max = max_distance(routes)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
        # Inter-route 2-opt* best improvement (one pass)
        improved = True
        while improved:
            improved = False
            best_improv = None
            best_new_max = float('inf')
            for t1 in range(truck_count):
                for t2 in range(t1+1, truck_count):
                    r1 = routes[t1]
                    r2 = routes[t2]
                    if len(r1) <= 2 or len(r2) <= 2:
                        continue
                    for i in range(1, len(r1)-1):
                        for j in range(1, len(r2)-1):
                            new_r1 = r1[:i+1] + r2[j+1:]
                            new_r2 = r2[:j+1] + r1[i+1:]
                            d1 = route_distance(new_r1)
                            d2 = route_distance(new_r2)
                            other_max = 0.0
                            for idx, r in enumerate(routes):
                                if idx not in (t1, t2):
                                    d = route_distance(r)
                                    if d > other_max:
                                        other_max = d
                            cand_max = max(d1, d2, other_max)
                            if cand_max < best_new_max - 1e-12:
                                best_new_max = cand_max
                                best_improv = (t1, t2, i, j, new_r1, new_r2)
            if best_improv is not None and best_new_max < cur_max - 1e-12:
                t1, t2, i, j, new_r1, new_r2 = best_improv
                routes[t1] = two_opt(new_r1)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
                improved = True
        # Max-route reduction: best improvement relocate and swap
        for _ in range(n):
            max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
            max_route = routes[max_idx]
            if len(max_route) <= 2:
                break
            best_improv = None
            best_new_max = float('inf')
            # best relocate
            for idx in range(1, len(max_route)-1):
                cust = max_route[idx]
                new_max_route = max_route[:idx] + max_route[idx+1:]
                for t2 in range(truck_count):
                    if t2 == max_idx:
                        continue
                    r2 = routes[t2]
                    for pos in range(1, len(r2)):
                        new_r2 = r2[:pos] + [cust] + r2[pos:]
                        d_max_new = route_distance(new_max_route)
                        d2_new = route_distance(new_r2)
                        other_max = 0.0
                        for idx2, r in enumerate(routes):
                            if idx2 not in (max_idx, t2):
                                d = route_distance(r)
                                if d > other_max:
                                    other_max = d
                        cand_max = max(d_max_new, d2_new, other_max)
                        if cand_max < best_new_max - 1e-12:
                            best_new_max = cand_max
                            best_improv = ('relocate', max_idx, idx, t2, pos, new_max_route, new_r2)
            # best swap
            for idx in range(1, len(max_route)-1):
                cust1 = max_route[idx]
                for t2 in range(truck_count):
                    if t2 == max_idx:
                        continue
                    r2 = routes[t2]
                    if len(r2) <= 2:
                        continue
                    for idx2 in range(1, len(r2)-1):
                        cust2 = r2[idx2]
                        new_max_route = max_route[:idx] + [cust2] + max_route[idx+1:]
                        new_r2 = r2[:idx2] + [cust1] + r2[idx2+1:]
                        d_max_new = route_distance(new_max_route)
                        d2_new = route_distance(new_r2)
                        other_max = 0.0
                        for idx3, r in enumerate(routes):
                            if idx3 not in (max_idx, t2):
                                d = route_distance(r)
                                if d > other_max:
                                    other_max = d
                        cand_max = max(d_max_new, d2_new, other_max)
                        if cand_max < best_new_max - 1e-12:
                            best_new_max = cand_max
                            best_improv = ('swap', max_idx, idx, t2, idx2, new_max_route, new_r2)
            if best_improv is not None and best_new_max < cur_max - 1e-12:
                if best_improv[0] == 'relocate':
                    _, max_idx, idx, t2, pos, new_max_route, new_r2 = best_improv
                else:
                    _, max_idx, idx, t2, idx2, new_max_route, new_r2 = best_improv
                routes[max_idx] = two_opt(new_max_route)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
            else:
                break
        # Split longest route: try to split the max route into two segments and distribute
        for _ in range(truck_count):
            max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
            max_route = routes[max_idx]
            if len(max_route) <= 4:
                break
            best_improv = None
            best_new_max = float('inf')
            # try all cut points between customers (excluding depots)
            for cut in range(1, len(max_route)-2):
                seg1 = max_route[:cut+1]  # from start depot through cut
                seg2 = max_route[cut+1:]  # from cut+1 to end depot
                # seg1 must start with 0 and end with some customer
                # seg2 must start with customer and end with 0
                # we need to insert seg1 into a truck and seg2 into another truck (could be same? but to reduce max, likely different)
                for t1 in range(truck_count):
                    if t1 == max_idx:
                        continue
                    for t2 in range(truck_count):
                        if t2 == max_idx or t2 == t1:
                            continue
                        r1 = routes[t1]
                        r2 = routes[t2]
                        # try all insertion positions for seg1 (except after depot?) actually segment already starts and ends with depot? No, seg1 starts with 0 and ends with a customer. We need to insert the whole seg1 as a route? But we are replacing the max route with seg1 and seg2? Better: we remove max route and assign seg1 to t1 and seg2 to t2. But we need to preserve depot endpoints. Actually, we can treat seg1 as a new route for t1 (replacing r1) and seg2 for t2. But we must keep depot endpoints. So we can set new_r1 = seg1 + [0]? Wait seg1 already ends with depot? No, max_route is [0,...,0], cut splits internal. So seg1 = [0,...,customer] and seg2 = [customer,...,0]. So if we assign seg1 to t1, we need to append 0? Actually seg1 already has 0 at start, but missing final 0. We can add 0 at end: new_r1 = seg1 + [0]. Similarly, seg2 already has 0 at end, but needs 0 at start? Actually seg2 starts with a customer, so we need to prepend 0: new_r2 = [0] + seg2. But that would make two depots in a row? Better: we can simply consider the segment as a complete route if we add depots appropriately. However, this might create double depot entries. Simpler: we can perform a 2-opt* style move: cut the max route at cut, and insert the two parts into other routes at the best positions. Let's do that: for each cut, we have two parts: part1 = max_route[1:cut+1] (customers only) and part2 = max_route[cut+1:-1] (customers only). Then we try all positions in t1 and t2 to insert part1 and part2 respectively. This is more general.
                # So we compute part1 = max_route[1:cut+1] (list of customers) and part2 = max_route[cut+1:-1]
                part1 = max_route[1:cut+1]
                part2 = max_route[cut+1:-1]
                if not part1 or not part2:
                    continue
                for t1 in range(truck_count):
                    if t1 == max_idx:
                        continue
                    r1 = routes[t1]
                    for pos1 in range(1, len(r1)):
                        new_r1 = r1[:pos1] + part1 + r1[pos1:]
                        for t2 in range(truck_count):
                            if t2 == max_idx or t2 == t1:
                                continue
                            r2 = routes[t2]
                            for pos2 in range(1, len(r2)):
                                new_r2 = r2[:pos2] + part2 + r2[pos2:]
                                # compute new max including new_r1, new_r2, and the unchanged routes except max_idx is removed
                                d1 = route_distance(new_r1)
                                d2 = route_distance(new_r2)
                                other_max = 0.0
                                for idx, r in enumerate(routes):
                                    if idx == max_idx:
                                        continue
                                    if idx == t1 or idx == t2:
                                        continue
                                    d = route_distance(r)
                                    if d > other_max:
                                        other_max = d
                                cand_max = max(d1, d2, other_max)
                                if cand_max < best_new_max - 1e-12:
                                    best_new_max = cand_max
                                    best_improv = (cut, t1, pos1, t2, pos2, new_r1, new_r2)
            if best_improv is not None and best_new_max < cur_max - 1e-12:
                cut, t1, pos1, t2, pos2, new_r1, new_r2 = best_improv
                # replace max_idx route with empty, then assign new_r1 and new_r2
                # Actually we need to keep truck count; we are reassigning customers from max_idx to t1 and t2, so max_idx becomes empty? Or we can keep max_idx as empty route [0,0] and use t1, t2. But we want to reduce max distance, so removing the max route is good.
                # We'll set routes[max_idx] = [0,0]
                routes[max_idx] = [0,0]
                routes[t1] = two_opt(new_r1)
                routes[t2] = two_opt(new_r2)
                cur_max = max_distance(routes)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [r[:] for r in routes]
                    report_best_vrp(best_routes)
            else:
                break
        # Perturbation: multiple random swaps (up to min(3, truck_count) swaps)
        num_swaps = min(3, truck_count)
        for _ in range(num_swaps):
            non_empty = [t for t in range(truck_count) if len(routes[t]) > 2]
            if len(non_empty) < 2:
                break
            t1, t2 = random.sample(non_empty, 2)
            r1 = routes[t1]
            r2 = routes[t2]
            i1 = random.randint(1, len(r1)-2)
            i2 = random.randint(1, len(r2)-2)
            cust1 = r1[i1]
            cust2 = r2[i2]
            new_r1 = r1[:i1] + [cust2] + r1[i1+1:]
            new_r2 = r2[:i2] + [cust1] + r2[i2+1:]
            routes[t1] = two_opt(new_r1)
            routes[t2] = two_opt(new_r2)
            cur_max = max_distance(routes)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
    return best_routes