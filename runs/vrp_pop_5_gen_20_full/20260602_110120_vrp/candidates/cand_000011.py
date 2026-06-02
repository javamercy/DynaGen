import numpy as np
import random
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = len(distance_matrix)
    # --- Construction: adaptive savings (from cand_000009) ---
    routes = []
    for i in range(1, n):
        routes.append([0, i, 0])
    while len(routes) < truck_count:
        routes.append([0, 0])
    if n == 1:
        return routes

    def route_dist(route):
        return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

    dist_cache = {}
    def get_dist(route):
        key = tuple(route)
        if key not in dist_cache:
            dist_cache[key] = route_dist(route)
        return dist_cache[key]

    while len(routes) > truck_count:
        candidates = []
        r_idx = list(range(len(routes)))
        current_max = max(get_dist(r) for r in routes)
        for i in r_idx:
            if len(routes[i]) <= 2:
                continue
            for j in r_idx:
                if i >= j or len(routes[j]) <= 2:
                    continue
                r1 = routes[i]
                r2 = routes[j]
                # merge r1 end to r2 start
                last1 = r1[-2]
                first2 = r2[1]
                savings = distance_matrix[0, last1] + distance_matrix[first2, 0] - distance_matrix[last1, first2]
                new_route = r1[:-1] + r2[1:]
                new_dist = get_dist(r1) + get_dist(r2) - distance_matrix[last1, 0] - distance_matrix[0, first2] + distance_matrix[last1, first2]
                new_max = max(new_dist, *[get_dist(routes[k]) for k in r_idx if k != i and k != j])
                threshold = current_max * 1.1
                if new_dist > threshold:
                    penalized_savings = savings * 0.5
                else:
                    penalized_savings = savings
                candidates.append((new_max, -penalized_savings, i, j, 0, last1, first2, new_route))
                # merge r2 end to r1 start
                last2 = r2[-2]
                first1 = r1[1]
                savings2 = distance_matrix[0, last2] + distance_matrix[first1, 0] - distance_matrix[last2, first1]
                new_route2 = r2[:-1] + r1[1:]
                new_dist2 = get_dist(r2) + get_dist(r1) - distance_matrix[last2, 0] - distance_matrix[0, first1] + distance_matrix[last2, first1]
                new_max2 = max(new_dist2, *[get_dist(routes[k]) for k in r_idx if k != i and k != j])
                if new_dist2 > threshold:
                    penalized_savings2 = savings2 * 0.5
                else:
                    penalized_savings2 = savings2
                candidates.append((new_max2, -penalized_savings2, i, j, 1, last2, first1, new_route2))
        if not candidates:
            break
        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        best = candidates[0]
        i, j = best[2], best[3]
        new_route = best[7]
        if i > j:
            i, j = j, i
        del routes[j]
        del routes[i]
        routes.append(new_route)
        dist_cache = {}

    # --- Multi-restart Simulated Annealing ---
    initial_routes = [r[:] for r in routes]
    num_restarts = 3
    best_overall_max = float('inf')
    best_overall_routes = None

    for restart in range(num_restarts):
        random.seed(restart * 9999)
        # perturb initial solution to diversify
        current_routes = [r[:] for r in initial_routes]
        for _ in range(n // 2):
            cust = random.randint(1, n-1)
            src_idx = None
            src_pos = None
            for idx, r in enumerate(current_routes):
                if cust in r:
                    src_idx = idx
                    src_pos = r.index(cust)
                    break
            if src_idx is None:
                continue
            tgt_idx = random.choice([i for i in range(len(current_routes)) if i != src_idx])
            max_pos = len(current_routes[tgt_idx]) - 1
            if max_pos >= 1:
                pos = random.randint(1, max_pos)
            else:
                pos = 1
            new_src = current_routes[src_idx][:src_pos] + current_routes[src_idx][src_pos+1:]
            if len(new_src) == 2:
                new_src = [0, 0]
            new_tgt = current_routes[tgt_idx][:pos] + [cust] + current_routes[tgt_idx][pos:]
            current_routes[src_idx] = new_src
            current_routes[tgt_idx] = new_tgt

        current_dists = [route_dist(r) for r in current_routes]
        current_max = max(current_dists)
        best_max_this = current_max
        best_routes_this = [r[:] for r in current_routes]

        # SA parameters
        max_iters = min(10000, n * 100)
        T0 = current_max * 0.1
        T_end = 1e-3
        alpha = (T_end / T0) ** (1.0 / max_iters) if max_iters > 0 else 0.9
        T = T0

        for it in range(max_iters):
            move_type = random.randint(0, 2)
            if move_type == 0:  # intra 2-opt
                feasible = [idx for idx, r in enumerate(current_routes) if len(r) > 3]
                if not feasible:
                    continue
                r_idx = random.choice(feasible)
                route = current_routes[r_idx]
                i = random.randint(1, len(route)-3)
                j = random.randint(i+1, len(route)-2)
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                new_dists = current_dists[:]
                new_dists[r_idx] = new_dist
                new_max = max(new_dists)
                delta = new_max - current_max
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_routes[r_idx] = new_route
                    current_dists = new_dists
                    current_max = new_max
                    if new_max < best_max_this:
                        best_max_this = new_max
                        best_routes_this = [r[:] for r in current_routes]
            elif move_type == 1:  # relocation
                cust = random.randint(1, n-1)
                src_idx = None
                src_pos = None
                for idx, r in enumerate(current_routes):
                    if cust in r:
                        src_idx = idx
                        src_pos = r.index(cust)
                        break
                if src_idx is None:
                    continue
                tgt_idx = random.choice([i for i in range(len(current_routes)) if i != src_idx])
                max_pos = len(current_routes[tgt_idx]) - 1
                if max_pos >= 1:
                    pos = random.randint(1, max_pos)
                else:
                    pos = 1
                new_src = current_routes[src_idx][:src_pos] + current_routes[src_idx][src_pos+1:]
                if len(new_src) == 2:
                    new_src = [0, 0]
                new_tgt = current_routes[tgt_idx][:pos] + [cust] + current_routes[tgt_idx][pos:]
                new_src_dist = sum(distance_matrix[new_src[k], new_src[k+1]] for k in range(len(new_src)-1))
                new_tgt_dist = sum(distance_matrix[new_tgt[k], new_tgt[k+1]] for k in range(len(new_tgt)-1))
                new_dists = current_dists[:]
                new_dists[src_idx] = new_src_dist
                new_dists[tgt_idx] = new_tgt_dist
                new_max = max(new_dists)
                delta = new_max - current_max
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_routes[src_idx] = new_src
                    current_routes[tgt_idx] = new_tgt
                    current_dists = new_dists
                    current_max = new_max
                    if new_max < best_max_this:
                        best_max_this = new_max
                        best_routes_this = [r[:] for r in current_routes]
            else:  # exchange
                idx1 = random.randint(0, len(current_routes)-1)
                idx2 = random.randint(0, len(current_routes)-1)
                if idx1 == idx2:
                    continue
                r1 = current_routes[idx1]
                r2 = current_routes[idx2]
                if len(r1) <= 2 or len(r2) <= 2:
                    continue
                pos1 = random.randint(1, len(r1)-2)
                pos2 = random.randint(1, len(r2)-2)
                cust1 = r1[pos1]
                cust2 = r2[pos2]
                new_r1 = r1[:pos1] + [cust2] + r1[pos1+1:]
                new_r2 = r2[:pos2] + [cust1] + r2[pos2+1:]
                new_dist1 = sum(distance_matrix[new_r1[k], new_r1[k+1]] for k in range(len(new_r1)-1))
                new_dist2 = sum(distance_matrix[new_r2[k], new_r2[k+1]] for k in range(len(new_r2)-1))
                new_dists = current_dists[:]
                new_dists[idx1] = new_dist1
                new_dists[idx2] = new_dist2
                new_max = max(new_dists)
                delta = new_max - current_max
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_routes[idx1] = new_r1
                    current_routes[idx2] = new_r2
                    current_dists = new_dists
                    current_max = new_max
                    if new_max < best_max_this:
                        best_max_this = new_max
                        best_routes_this = [r[:] for r in current_routes]
            T *= alpha

        if best_max_this < best_overall_max:
            best_overall_max = best_max_this
            best_overall_routes = [r[:] for r in best_routes_this]

    routes = [r[:] for r in best_overall_routes]
    # --- Final deterministic local search (from cand_000009) ---
    dist_cache.clear()
    for r in routes:
        dist_cache[tuple(r)] = route_dist(r)

    max_iter = n
    for _ in range(max_iter):
        improved = False
        # Intra-route 2-opt
        for idx, route in enumerate(routes):
            if len(route) <= 3:
                continue
            best_route = route[:]
            best_dist = get_dist(best_route)
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_dist = sum(distance_matrix[new_route[k], new_route[k+1]] for k in range(len(new_route)-1))
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_route = new_route[:]
                        improved = True
            routes[idx] = best_route
            dist_cache[tuple(best_route)] = best_dist

        # Inter-route relocation focused on reducing max
        route_dists = [get_dist(r) for r in routes]
        current_max = max(route_dists)
        for cust in range(1, n):
            src_idx = None
            src_pos = None
            for idx, route in enumerate(routes):
                if cust in route:
                    src_idx = idx
                    src_pos = route.index(cust)
                    break
            if src_idx is None:
                continue
            src_route = routes[src_idx]
            new_src = src_route[:src_pos] + src_route[src_pos+1:]
            if len(new_src) == 2:
                new_src = [0, 0]
            new_src_dist = sum(distance_matrix[new_src[k], new_src[k+1]] for k in range(len(new_src)-1))
            for tgt_idx, tgt_route in enumerate(routes):
                if tgt_idx == src_idx:
                    continue
                if len(tgt_route) <= 2:
                    new_tgt = [0, cust, 0]
                    new_tgt_dist = distance_matrix[0, cust] + distance_matrix[cust, 0]
                    new_max = max(new_src_dist, new_tgt_dist, *[d for i, d in enumerate(route_dists) if i not in (src_idx, tgt_idx)])
                    if new_max < current_max:
                        routes[src_idx] = new_src
                        routes[tgt_idx] = new_tgt
                        dist_cache[tuple(new_src)] = new_src_dist
                        dist_cache[tuple(new_tgt)] = new_tgt_dist
                        current_max = new_max
                        route_dists[src_idx] = new_src_dist
                        route_dists[tgt_idx] = new_tgt_dist
                        improved = True
                        break
                else:
                    for pos in range(1, len(tgt_route)):
                        new_tgt = tgt_route[:pos] + [cust] + tgt_route[pos:]
                        new_tgt_dist = sum(distance_matrix[new_tgt[k], new_tgt[k+1]] for k in range(len(new_tgt)-1))
                        new_max = max(new_src_dist, new_tgt_dist, *[d for i, d in enumerate(route_dists) if i not in (src_idx, tgt_idx)])
                        if new_max < current_max:
                            routes[src_idx] = new_src
                            routes[tgt_idx] = new_tgt
                            dist_cache[tuple(new_src)] = new_src_dist
                            dist_cache[tuple(new_tgt)] = new_tgt_dist
                            current_max = new_max
                            route_dists[src_idx] = new_src_dist
                            route_dists[tgt_idx] = new_tgt_dist
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        if not improved:
            break

    report_best_vrp(routes)
    return routes