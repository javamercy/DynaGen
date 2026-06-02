import numpy as np
import random
import math

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

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def savings_construction(seed):
        random.seed(seed)
        # Initialize each customer on its own route
        routes = [[0, c, 0] for c in customers]
        # Compute savings
        savings = []
        for i in customers:
            for j in customers:
                if i < j:
                    s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                    savings.append((s, i, j))
        savings.sort(key=lambda x: -x[0])
        used = set()
        # Merging loops
        for s, i, j in savings:
            if i in used or j in used:
                continue
            # Find routes containing i and j
            ri = rj = None
            for idx, r in enumerate(routes):
                if i in r and len(r) >= 3:
                    ri = idx
                if j in r and len(r) >= 3:
                    rj = idx
            if ri is None or rj is None:
                continue
            if ri == rj:
                continue
            # Check if merging is possible (i and j are endpoints)
            r_i = routes[ri]
            r_j = routes[rj]
            # Check if i is at the end (just before 0) and j is at the start (just after 0)
            if r_i[-2] == i and r_j[1] == j:
                new_route = r_i[:-1] + r_j[1:]
                if len(new_route) - 2 <= n - (truck_count - 1):  # enough capacity? not needed, just ensure we don't exceed max routes? We'll allow any number of routes initially, then later reduce.
                    routes[ri] = new_route
                    routes[rj] = [0, 0]
                    used.add(i)
                    used.add(j)
            elif r_i[1] == i and r_j[-2] == j:
                new_route = r_j[:-1] + r_i[1:]
                routes[ri] = new_route
                routes[rj] = [0, 0]
                used.add(i)
                used.add(j)
            elif r_i[-2] == i and r_j[-2] == j:
                new_route = r_i[:-1] + r_j[-2::-1] + [0]
                routes[ri] = new_route
                routes[rj] = [0, 0]
                used.add(i)
                used.add(j)
            elif r_i[1] == i and r_j[1] == j:
                new_route = [0] + r_j[1:-1][::-1] + r_i[1:]
                routes[ri] = new_route
                routes[rj] = [0, 0]
                used.add(i)
                used.add(j)
            # else cannot merge
        # Remove empty routes
        routes = [r for r in routes if len(r) > 2]
        # If we have fewer routes than truck_count, split longest routes
        while len(routes) < truck_count:
            # Split the longest route
            longest_idx = max(range(len(routes)), key=lambda i: route_distance(routes[i]))
            longest = routes[longest_idx]
            if len(longest) <= 3:
                break
            # Find best split point (minimize max of two subsequences)
            best_split = None
            best_max = float('inf')
            for k in range(1, len(longest)-2):
                r1 = longest[:k+1] + [0]
                r2 = [0] + longest[k+1:]
                d1 = route_distance(r1)
                d2 = route_distance(r2)
                m = max(d1, d2)
                if m < best_max:
                    best_max = m
                    best_split = k
            if best_split is None:
                break
            k = best_split
            r1 = longest[:k+1] + [0]
            r2 = [0] + longest[k+1:]
            routes[longest_idx] = r1
            routes.append(r2)
        # If we have more routes than truck_count, merge some
        while len(routes) > truck_count:
            # Find two routes to merge that minimize max distance
            best_pair = None
            best_max = float('inf')
            for i in range(len(routes)):
                for j in range(i+1, len(routes)):
                    # Try each orientation
                    for rev_i, rev_j in [(False, False), (False, True), (True, False), (True, True)]:
                        ri = routes[i][1:-1] if not rev_i else routes[i][1:-1][::-1]
                        rj = routes[j][1:-1] if not rev_j else routes[j][1:-1][::-1]
                        merged = [0] + ri + rj + [0]
                        d = route_distance(merged)
                        if d < best_max:
                            best_max = d
                            best_pair = (i, j, rev_i, rev_j)
            if best_pair is None:
                break
            i, j, rev_i, rev_j = best_pair
            ri = routes[i][1:-1] if not rev_i else routes[i][1:-1][::-1]
            rj = routes[j][1:-1] if not rev_j else routes[j][1:-1][::-1]
            merged = [0] + ri + rj + [0]
            routes[i] = merged
            routes.pop(j)
        # Now routes length should be exactly truck_count (or less, pad with empties)
        while len(routes) < truck_count:
            routes.append([0, 0])
        return routes

    best_routes = None
    best_max = float('inf')
    max_restarts = max(truck_count, 10)
    for restart in range(max_restarts):
        routes = savings_construction(restart)
        # Local search
        for _ in range(n):  # multiple passes
            # Intra-route 2-opt
            for t in range(truck_count):
                routes[t] = two_opt(routes[t])
            cur_max = max_distance(routes)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [r[:] for r in routes]
                report_best_vrp(best_routes)
            # Inter-route 2-opt* best improvement
            improved = True
            while improved:
                improved = False
                best_move = None
                best_new_max = cur_max
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
                                    best_move = (t1, t2, i, j, new_r1, new_r2)
                if best_move is not None and best_new_max < cur_max - 1e-12:
                    t1, t2, i, j, new_r1, new_r2 = best_move
                    routes[t1] = two_opt(new_r1)
                    routes[t2] = two_opt(new_r2)
                    cur_max = max_distance(routes)
                    if cur_max < best_max - 1e-12:
                        best_max = cur_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                    improved = True
            # Greedy reduction of longest route via relocate
            for _ in range(n):
                max_idx = max(range(truck_count), key=lambda t: route_distance(routes[t]))
                max_route = routes[max_idx]
                if len(max_route) <= 2:
                    break
                found = False
                best_relocate = None
                best_relocate_max = float('inf')
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
                            if cand_max < best_relocate_max - 1e-12:
                                best_relocate_max = cand_max
                                best_relocate = (t2, pos, idx, new_max_route, new_r2)
                if best_relocate is not None and best_relocate_max < cur_max - 1e-12:
                    t2, pos, idx, new_max_route, new_r2 = best_relocate
                    routes[max_idx] = two_opt(new_max_route)
                    routes[t2] = two_opt(new_r2)
                    cur_max = max_distance(routes)
                    if cur_max < best_max - 1e-12:
                        best_max = cur_max
                        best_routes = [r[:] for r in routes]
                        report_best_vrp(best_routes)
                    found = True
                if not found:
                    break
        # Perturbation: double-bridge on a random route
        non_empty = [t for t in range(truck_count) if len(routes[t]) > 6]
        if non_empty:
            t = random.choice(non_empty)
            route = routes[t]
            if len(route) >= 8:
                # Choose 3 cut points
                a = random.randint(1, len(route)-3)
                b = random.randint(a+1, len(route)-2)
                c = random.randint(b+1, len(route)-1)
                # Segment order: A, B, C, D
                A = route[1:a+1]
                B = route[a+1:b+1]
                C = route[b+1:c+1]
                D = route[c+1:-1]
                # New order: A, C, B, D
                new_route = [0] + A + C + B + D + [0]
                routes[t] = two_opt(new_route)
        # Check if perturbation improved
        cur_max = max_distance(routes)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [r[:] for r in routes]
            report_best_vrp(best_routes)
    return best_routes