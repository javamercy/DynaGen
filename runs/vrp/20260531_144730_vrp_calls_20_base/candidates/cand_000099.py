import numpy as np
import random
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []

    def report_best_vrp(routes):
        pass

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def best_insertion(c, routes, route_dists):
        best = (float('inf'), -1, -1)
        second = (float('inf'), -1, -1)
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            other_max = 0.0
            for j, d in enumerate(route_dists):
                if j != r_idx and d > other_max:
                    other_max = d
            for pos in range(1, len(route)):
                pred = route[pos-1]
                succ = route[pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best, second = (new_max, r_idx, pos), best
                elif new_max < second[0]:
                    second = (new_max, r_idx, pos)
        return best[0], best[1], best[2], second[0]

    # Initial construction using regret heuristic
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    while unassigned:
        bests = []
        for c in unassigned:
            best_new_max, best_route, best_pos, second_new_max = best_insertion(c, routes, route_dists)
            if best_route == -1:
                continue
            regret = second_new_max - best_new_max if second_new_max != float('inf') else float('inf')
            bests.append((-regret, c, best_route, best_pos, best_new_max))
        bests.sort(key=lambda x: (x[0], x[1]))
        _, c, best_route, best_pos, new_max = bests[0]
        route = routes[best_route]
        route.insert(best_pos, c)
        route_dists[best_route] = route_dist(route)
        unassigned.remove(c)
        report_best_vrp(routes)

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)

    # Intra-route 2-opt until local optimum
    for r_idx in range(truck_count):
        improved = True
        while improved:
            improved = False
            route = routes[r_idx]
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    old = distance_matrix[route[i-1], route[i]] + distance_matrix[route[j], route[j+1]]
                    new = distance_matrix[route[i-1], route[j]] + distance_matrix[route[i], route[j+1]]
                    if new < old - 1e-12:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        route_dists[r_idx] = route_dist(route)
                        break
                if improved:
                    break
        report_best_vrp(routes)
        cur_max = max(route_dists)
        if cur_max < best_max - 1e-12:
            best_max = cur_max
            best_routes = [route[:] for route in routes]

    # Intensive improvement: repeat relocate, swap, 2-opt* until no improvement
    max_iter = n * truck_count
    for _ in range(max_iter):
        improved_overall = False
        # Best-improvement relocate from longest route
        max_dist = max(route_dists)
        max_idx = route_dists.index(max_dist)
        best_move = None
        best_new_max = max_dist
        route_max = routes[max_idx]
        for i in range(1, len(route_max)-1):
            c = route_max[i]
            pred = route_max[i-1]
            succ = route_max[i+1]
            new_max_dist = route_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for pos in range(1, len(other_route)):
                    pred_o = other_route[pos-1]
                    succ_o = other_route[pos]
                    new_other = route_dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                    other_max = 0.0
                    for j, d in enumerate(route_dists):
                        if j != max_idx and j != other_idx and d > other_max:
                            other_max = d
                    new_overall = max(other_max, new_max_dist, new_other)
                    if new_overall < best_new_max - 1e-12:
                        best_new_max = new_overall
                        best_move = (i, other_idx, pos, new_max_dist, new_other)
        if best_move is not None:
            i, other_idx, pos, new_max_dist, new_other = best_move
            c = route_max.pop(i)
            routes[other_idx].insert(pos, c)
            route_dists[max_idx] = new_max_dist
            route_dists[other_idx] = new_other
            for r_idx in [max_idx, other_idx]:
                improved = True
                while improved:
                    improved = False
                    route = routes[r_idx]
                    for a in range(1, len(route)-2):
                        for b in range(a+1, len(route)-1):
                            old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                            new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                            if new < old - 1e-12:
                                route[a:b+1] = reversed(route[a:b+1])
                                improved = True
                                route_dists[r_idx] = route_dist(route)
                                break
                        if improved:
                            break
            report_best_vrp(routes)
            cur_max = max(route_dists)
            if cur_max < best_max - 1e-12:
                best_max = cur_max
                best_routes = [route[:] for route in routes]
            improved_overall = True

        # Best-improvement swap
        if not improved_overall:
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_swap = None
            best_new_max = max_dist
            route_max = routes[max_idx]
            for i in range(1, len(route_max)-1):
                c1 = route_max[i]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for j in range(1, len(other_route)-1):
                        c2 = other_route[j]
                        old1 = route_dists[max_idx]
                        old2 = route_dists[other_idx]
                        pred1 = route_max[i-1]
                        succ1 = route_max[i+1]
                        new_dist_max = old1 - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                        pred2 = other_route[j-1]
                        succ2 = other_route[j+1]
                        new_dist_other = old2 - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                        other_max = 0.0
                        for k, d in enumerate(route_dists):
                            if k != max_idx and k != other_idx and d > other_max:
                                other_max = d
                        new_overall = max(other_max, new_dist_max, new_dist_other)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
            if best_swap is not None:
                i, other_idx, j, new_dist_max, new_dist_other = best_swap
                route_max = routes[max_idx]
                other_route = routes[other_idx]
                c1 = route_max[i]
                c2 = other_route[j]
                route_max[i] = c2
                other_route[j] = c1
                route_dists[max_idx] = new_dist_max
                route_dists[other_idx] = new_dist_other
                for r_idx in [max_idx, other_idx]:
                    improved = True
                    while improved:
                        improved = False
                        route = routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved = True
                                    route_dists[r_idx] = route_dist(route)
                                    break
                            if improved:
                                break
                report_best_vrp(routes)
                cur_max = max(route_dists)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [route[:] for route in routes]
                improved_overall = True

        # Best-improvement 2-opt* (swap suffixes)
        if not improved_overall:
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_cross = None
            best_new_max = max_dist
            route_max = routes[max_idx]
            for other_idx in range(truck_count):
                if other_idx == max_idx:
                    continue
                other_route = routes[other_idx]
                for i in range(1, len(route_max)-1):
                    for j in range(1, len(other_route)-1):
                        if route_max[-1] != 0 or other_route[-1] != 0:
                            continue
                        old1 = distance_matrix[route_max[i], route_max[i+1]]
                        old2 = distance_matrix[other_route[j], other_route[j+1]]
                        new1 = distance_matrix[route_max[i], other_route[j+1]]
                        new2 = distance_matrix[other_route[j], route_max[i+1]]
                        new_dist_max = route_dists[max_idx] - old1 + new1
                        new_dist_other = route_dists[other_idx] - old2 + new2
                        other_max = 0.0
                        for k, d in enumerate(route_dists):
                            if k != max_idx and k != other_idx and d > other_max:
                                other_max = d
                        new_overall = max(other_max, new_dist_max, new_dist_other)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
            if best_cross is not None:
                i, other_idx, j, new_dist_max, new_dist_other = best_cross
                route_max = routes[max_idx]
                other_route = routes[other_idx]
                new_route_max = route_max[:i+1] + other_route[j+1:]
                new_route_other = other_route[:j+1] + route_max[i+1:]
                routes[max_idx] = new_route_max
                routes[other_idx] = new_route_other
                route_dists[max_idx] = route_dist(new_route_max)
                route_dists[other_idx] = route_dist(new_route_other)
                for r_idx in [max_idx, other_idx]:
                    improved = True
                    while improved:
                        improved = False
                        route = routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved = True
                                    route_dists[r_idx] = route_dist(route)
                                    break
                            if improved:
                                break
                report_best_vrp(routes)
                cur_max = max(route_dists)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [route[:] for route in routes]
                improved_overall = True

        if not improved_overall:
            break

    # Iterated local search phase with adaptive perturbation (replaced with double-bridge)
    outer_iterations = min(50, n * 2)
    stagnation_limit = max(5, n // 20)
    max_perturbation = max(5, n // 10)
    last_improvement = 0
    perturbation_strength = 1
    for iteration in range(outer_iterations):
        # Start from best solution
        routes = [route[:] for route in best_routes]
        route_dists = [route_dist(r) for r in routes]
        # Adapt perturbation strength on stagnation
        if iteration - last_improvement > stagnation_limit:
            perturbation_strength = min(perturbation_strength + 1, max_perturbation)
        # Perturb using double-bridge move(s) on the longest route
        for _ in range(perturbation_strength):
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            route = routes[max_idx]
            if len(route) < 6:  # need at least depot, 4 customers, depot
                # Fall back to random relocate
                if len(route) <= 2:
                    break
                i = random.randint(1, len(route)-2)
                c = route.pop(i)
                candidates = [r for r in range(truck_count) if r != max_idx]
                other_idx = random.choice(candidates)
                other_route = routes[other_idx]
                pos = random.randint(1, len(other_route)-1)
                other_route.insert(pos, c)
                route_dists[max_idx] = route_dist(route)
                route_dists[other_idx] = route_dist(other_route)
                continue
            # Perform double-bridge: cut the route at positions a,b,c,d ensuring non-overlapping segments
            # We select a random segment and reinsert it elsewhere
            # Simple implementation: cut at two random points, swap the two resulting segments
            L = len(route)
            # Choose two cut positions (1..L-2 to ensure at least one customer in each segment? Actually we need at least 1 customer per segment after splits)
            p1 = random.randint(1, L-4)
            p2 = random.randint(p1+2, L-2)
            # Splits: [1:p1], [p1+1:p2], [p2+1:-1] where -1 is last depot
            # Reorder: first segment, third, second, then original order? Or swap two segments? Standard double-bridge swaps two pairs of edges. To keep it simple, we'll take the middle segment (p1+1:p2+1) and move it to after the second segment? Actually we'll cut after p1 and p2, then swap the parts between them.
            # I'll do: new_route = route[:p1+1] + route[p2+1:-1] + route[p1+1:p2+1] + [0] (keeping depot) but that changes start/end. Since route starts and ends with 0, we keep the start 0 and end 0.
            # Better: choose two points a<b<c? Use three cuts? Simpler: do a 2-opt-like swap but cross two routes? No, double bridge is within a route.
            # We'll implement a simple version: break the route into three parts by random indices i and j (i<j), then rearrange as: [0 to i], [j+1 to end], [i+1 to j] (swap middle and end). This is a partial reversal? Actually it's a type of double-bridge.
            i = random.randint(1, L-3)
            j = random.randint(i+1, L-2)
            # new order: start, segment1 (i+1..j), segment2 (j+1..-1? excluding last depot?), then rest? Actually we want to keep depots fixed. So we exchange the subroutes after i and j.
            # Standard double-bridge: choose four points, exchange pairs. For simplicity and to avoid errors, we'll do a random permutation of three segments that are nonempty and keep depot positions.
            # Let's do: cut at i and j (0 < i < j < L-1). Segments: A = route[1:i+1], B = route[i+1:j+1], C = route[j+1:-1]. Reassemble as [0] + random permutation of [A,B,C] + [0]. Choose random order.
            segments = [route[1:i+1], route[i+1:j+1], route[j+1:-1]]
            # Ensure each segment has at least 1 node? Possibly, but even if empty, it's ok.
            random.shuffle(segments)
            new_route = [0] + segments[0] + segments[1] + segments[2] + [0]
            # But this might change the set of customers? No, it's a permutation.
            routes[max_idx] = new_route
            route_dists[max_idx] = route_dist(new_route)
        # Apply intra-route 2-opt on all routes after perturbation
        for r_idx in range(truck_count):
            improved = True
            while improved:
                improved = False
                route = routes[r_idx]
                for a in range(1, len(route)-2):
                    for b in range(a+1, len(route)-1):
                        old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                        new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                        if new < old - 1e-12:
                            route[a:b+1] = reversed(route[a:b+1])
                            improved = True
                            route_dists[r_idx] = route_dist(route)
                            break
                    if improved:
                        break
        # Then re-apply the full improvement loop
        max_iter = n * truck_count
        for _ in range(max_iter):
            improved_overall = False
            # Best-improvement relocate
            max_dist = max(route_dists)
            max_idx = route_dists.index(max_dist)
            best_move = None
            best_new_max = max_dist
            route_max = routes[max_idx]
            for i in range(1, len(route_max)-1):
                c = route_max[i]
                pred = route_max[i-1]
                succ = route_max[i+1]
                new_max_dist = route_dists[max_idx] - distance_matrix[pred, c] - distance_matrix[c, succ] + distance_matrix[pred, succ]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(1, len(other_route)):
                        pred_o = other_route[pos-1]
                        succ_o = other_route[pos]
                        new_other = route_dists[other_idx] - distance_matrix[pred_o, succ_o] + distance_matrix[pred_o, c] + distance_matrix[c, succ_o]
                        other_max = 0.0
                        for j, d in enumerate(route_dists):
                            if j != max_idx and j != other_idx and d > other_max:
                                other_max = d
                        new_overall = max(other_max, new_max_dist, new_other)
                        if new_overall < best_new_max - 1e-12:
                            best_new_max = new_overall
                            best_move = (i, other_idx, pos, new_max_dist, new_other)
            if best_move is not None:
                i, other_idx, pos, new_max_dist, new_other = best_move
                c = route_max.pop(i)
                routes[other_idx].insert(pos, c)
                route_dists[max_idx] = new_max_dist
                route_dists[other_idx] = new_other
                for r_idx in [max_idx, other_idx]:
                    improved = True
                    while improved:
                        improved = False
                        route = routes[r_idx]
                        for a in range(1, len(route)-2):
                            for b in range(a+1, len(route)-1):
                                old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                if new < old - 1e-12:
                                    route[a:b+1] = reversed(route[a:b+1])
                                    improved = True
                                    route_dists[r_idx] = route_dist(route)
                                    break
                            if improved:
                                break
                cur_max = max(route_dists)
                if cur_max < best_max - 1e-12:
                    best_max = cur_max
                    best_routes = [route[:] for route in routes]
                    report_best_vrp(routes)
                    last_improvement = iteration
                    perturbation_strength = 1
                improved_overall = True

            # Best-improvement swap
            if not improved_overall:
                max_dist = max(route_dists)
                max_idx = route_dists.index(max_dist)
                best_swap = None
                best_new_max = max_dist
                route_max = routes[max_idx]
                for i in range(1, len(route_max)-1):
                    c1 = route_max[i]
                    for other_idx in range(truck_count):
                        if other_idx == max_idx:
                            continue
                        other_route = routes[other_idx]
                        for j in range(1, len(other_route)-1):
                            c2 = other_route[j]
                            old1 = route_dists[max_idx]
                            old2 = route_dists[other_idx]
                            pred1 = route_max[i-1]
                            succ1 = route_max[i+1]
                            new_dist_max = old1 - distance_matrix[pred1, c1] - distance_matrix[c1, succ1] + distance_matrix[pred1, c2] + distance_matrix[c2, succ1]
                            pred2 = other_route[j-1]
                            succ2 = other_route[j+1]
                            new_dist_other = old2 - distance_matrix[pred2, c2] - distance_matrix[c2, succ2] + distance_matrix[pred2, c1] + distance_matrix[c1, succ2]
                            other_max = 0.0
                            for k, d in enumerate(route_dists):
                                if k != max_idx and k != other_idx and d > other_max:
                                    other_max = d
                            new_overall = max(other_max, new_dist_max, new_dist_other)
                            if new_overall < best_new_max - 1e-12:
                                best_new_max = new_overall
                                best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
                if best_swap is not None:
                    i, other_idx, j, new_dist_max, new_dist_other = best_swap
                    route_max = routes[max_idx]
                    other_route = routes[other_idx]
                    c1 = route_max[i]
                    c2 = other_route[j]
                    route_max[i] = c2
                    other_route[j] = c1
                    route_dists[max_idx] = new_dist_max
                    route_dists[other_idx] = new_dist_other
                    for r_idx in [max_idx, other_idx]:
                        improved = True
                        while improved:
                            improved = False
                            route = routes[r_idx]
                            for a in range(1, len(route)-2):
                                for b in range(a+1, len(route)-1):
                                    old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                    new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                    if new < old - 1e-12:
                                        route[a:b+1] = reversed(route[a:b+1])
                                        improved = True
                                        route_dists[r_idx] = route_dist(route)
                                        break
                                if improved:
                                    break
                    cur_max = max(route_dists)
                    if cur_max < best_max - 1e-12:
                        best_max = cur_max
                        best_routes = [route[:] for route in routes]
                        report_best_vrp(routes)
                        last_improvement = iteration
                        perturbation_strength = 1
                    improved_overall = True

            # Best-improvement 2-opt*
            if not improved_overall:
                max_dist = max(route_dists)
                max_idx = route_dists.index(max_dist)
                best_cross = None
                best_new_max = max_dist
                route_max = routes[max_idx]
                for other_idx in range(truck_count):
                    if other_idx == max_idx:
                        continue
                    other_route = routes[other_idx]
                    for i in range(1, len(route_max)-1):
                        for j in range(1, len(other_route)-1):
                            if route_max[-1] != 0 or other_route[-1] != 0:
                                continue
                            old1 = distance_matrix[route_max[i], route_max[i+1]]
                            old2 = distance_matrix[other_route[j], other_route[j+1]]
                            new1 = distance_matrix[route_max[i], other_route[j+1]]
                            new2 = distance_matrix[other_route[j], route_max[i+1]]
                            new_dist_max = route_dists[max_idx] - old1 + new1
                            new_dist_other = route_dists[other_idx] - old2 + new2
                            other_max = 0.0
                            for k, d in enumerate(route_dists):
                                if k != max_idx and k != other_idx and d > other_max:
                                    other_max = d
                            new_overall = max(other_max, new_dist_max, new_dist_other)
                            if new_overall < best_new_max - 1e-12:
                                best_new_max = new_overall
                                best_cross = (i, other_idx, j, new_dist_max, new_dist_other)
                if best_cross is not None:
                    i, other_idx, j, new_dist_max, new_dist_other = best_cross
                    route_max = routes[max_idx]
                    other_route = routes[other_idx]
                    new_route_max = route_max[:i+1] + other_route[j+1:]
                    new_route_other = other_route[:j+1] + route_max[i+1:]
                    routes[max_idx] = new_route_max
                    routes[other_idx] = new_route_other
                    route_dists[max_idx] = route_dist(new_route_max)
                    route_dists[other_idx] = route_dist(new_route_other)
                    for r_idx in [max_idx, other_idx]:
                        improved = True
                        while improved:
                            improved = False
                            route = routes[r_idx]
                            for a in range(1, len(route)-2):
                                for b in range(a+1, len(route)-1):
                                    old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                                    new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                                    if new < old - 1e-12:
                                        route[a:b+1] = reversed(route[a:b+1])
                                        improved = True
                                        route_dists[r_idx] = route_dist(route)
                                        break
                                if improved:
                                    break
                    cur_max = max(route_dists)
                    if cur_max < best_max - 1e-12:
                        best_max = cur_max
                        best_routes = [route[:] for route in routes]
                        report_best_vrp(routes)
                        last_improvement = iteration
                        perturbation_strength = 1
                    improved_overall = True

            if not improved_overall:
                break

        # Check if we are stuck
        if iteration - last_improvement > stagnation_limit * 3:
            break

    return best_routes