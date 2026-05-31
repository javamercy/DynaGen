import numpy as np
import random

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    if truck_count <= 0:
        return []
    random.seed(0)

    def route_dist(route):
        d = 0.0
        for i in range(len(route)-1):
            d += distance_matrix[route[i], route[i+1]]
        return d

    def total_dist(routes):
        return sum(route_dist(r) for r in routes)

    # Initial construction: cheapest insertion minimizing max route distance
    routes = [[0, 0] for _ in range(truck_count)]
    route_dists = [0.0] * truck_count
    unassigned = list(range(1, n))
    random.shuffle(unassigned)
    for c in unassigned:
        best = (float('inf'), -1, -1)
        for r_idx in range(truck_count):
            for pos in range(1, len(routes[r_idx])):
                pred = routes[r_idx][pos-1]
                succ = routes[r_idx][pos]
                new_dist = route_dists[r_idx] - distance_matrix[pred, succ] + distance_matrix[pred, c] + distance_matrix[c, succ]
                # compute max if we insert here
                other_max = 0.0
                for j, d in enumerate(route_dists):
                    if j != r_idx and d > other_max:
                        other_max = d
                new_max = max(other_max, new_dist)
                if new_max < best[0]:
                    best = (new_max, r_idx, pos)
        _, r_idx, pos = best
        routes[r_idx].insert(pos, c)
        route_dists[r_idx] = route_dist(routes[r_idx])

    best_routes = [route[:] for route in routes]
    best_max = max(route_dists)
    best_total = total_dist(routes)
    report_best_vrp(best_routes)

    # Improvement loop
    max_iter = 50 * n  # finite bound
    for _ in range(max_iter):
        improved = False
        # Find route with current max distance
        max_dist = max(route_dists)
        max_idx = route_dists.index(max_dist)
        route_max = routes[max_idx]

        # Relocate customer from max route to other routes
        best_reloc = None
        best_new_max = max_dist
        best_new_total = total_dist(routes)
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
                        best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_max_dist + new_other
                        best_reloc = (i, other_idx, pos, new_max_dist, new_other)
                    elif abs(new_overall - best_new_max) < 1e-12:
                        new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_max_dist + new_other
                        if new_total < best_new_total - 1e-12:
                            best_new_max = new_overall
                            best_new_total = new_total
                            best_reloc = (i, other_idx, pos, new_max_dist, new_other)

        # Swap customers between max route and others
        best_swap = None
        for other_idx in range(truck_count):
            if other_idx == max_idx:
                continue
            other_route = routes[other_idx]
            for i in range(1, len(route_max)-1):
                for j in range(1, len(other_route)-1):
                    c1 = route_max[i]
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
                        best_new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                        best_swap = (i, other_idx, j, new_dist_max, new_dist_other)
                    elif abs(new_overall - best_new_max) < 1e-12:
                        new_total = best_total - route_dists[max_idx] - route_dists[other_idx] + new_dist_max + new_dist_other
                        if new_total < best_new_total - 1e-12:
                            best_new_max = new_overall
                            best_new_total = new_total
                            best_swap = (i, other_idx, j, new_dist_max, new_dist_other)

        # Apply best move if improves max or total
        if best_reloc is not None and (best_new_max < best_max - 1e-12 or (abs(best_new_max - best_max) < 1e-12 and best_new_total < best_total - 1e-12)):
            i, other_idx, pos, new_max_dist, new_other = best_reloc
            c = route_max.pop(i)
            routes[other_idx].insert(pos, c)
            route_dists[max_idx] = new_max_dist
            route_dists[other_idx] = new_other
            improved = True
        elif best_swap is not None and (best_new_max < best_max - 1e-12 or (abs(best_new_max - best_max) < 1e-12 and best_new_total < best_total - 1e-12)):
            i, other_idx, j, new_dist_max, new_dist_other = best_swap
            c1 = route_max[i]
            c2 = routes[other_idx][j]
            route_max[i] = c2
            routes[other_idx][j] = c1
            route_dists[max_idx] = new_dist_max
            route_dists[other_idx] = new_dist_other
            improved = True

        if improved:
            # Intra-route 2-opt on affected routes
            for r_idx in [max_idx, other_idx]:
                route = routes[r_idx]
                improved_inner = True
                while improved_inner:
                    improved_inner = False
                    for a in range(1, len(route)-2):
                        for b in range(a+1, len(route)-1):
                            old = distance_matrix[route[a-1], route[a]] + distance_matrix[route[b], route[b+1]]
                            new = distance_matrix[route[a-1], route[b]] + distance_matrix[route[a], route[b+1]]
                            if new < old - 1e-12:
                                route[a:b+1] = reversed(route[a:b+1])
                                improved_inner = True
                                route_dists[r_idx] = route_dist(route)
                                break
                        if improved_inner:
                            break
            # Update best
            cur_max = max(route_dists)
            cur_total = total_dist(routes)
            if cur_max < best_max - 1e-12 or (abs(cur_max - best_max) < 1e-12 and cur_total < best_total - 1e-12):
                best_max = cur_max
                best_total = cur_total
                best_routes = [route[:] for route in routes]
                report_best_vrp(best_routes)
        else:
            # No improving move found, check swap that doesn't change max but may reduce total
            # Not needed, break if no improvement
            break

    return best_routes