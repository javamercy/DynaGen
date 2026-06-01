import numpy as np
import heapq
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    num_customers = n - 1

    # Construction: each customer as separate route
    routes = [[0, i, 0] for i in range(1, n)]
    for _ in range(truck_count - num_customers):
        routes.append([0, 0])

    if num_customers > 1:
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                savings.append((-s, i, j))
        heapq.heapify(savings)

        # Map customer to route index
        customer_route = {}
        route_ends = {}
        for idx, route in enumerate(routes):
            for c in route[1:-1]:
                customer_route[c] = idx
            if len(route) > 2:
                route_ends[idx] = (route[1], route[-2])
            else:
                route_ends[idx] = ()

        while len(routes) > truck_count and savings:
            neg_s, i, j = heapq.heappop(savings)
            if i not in customer_route or j not in customer_route:
                continue
            ri = customer_route[i]
            rj = customer_route[j]
            if ri == rj:
                continue
            ends_i = route_ends[ri]
            ends_j = route_ends[rj]
            if not ends_i or not ends_j:
                continue
            if i not in ends_i or j not in ends_j:
                continue
            # Determine merge orientation
            inner_i = [c for c in routes[ri][1:-1]]
            inner_j = [c for c in routes[rj][1:-1]]
            combo = None
            if i == inner_i[-1] and j == inner_j[0]:
                combo = inner_i + inner_j
            elif i == inner_i[-1] and j == inner_j[-1]:
                combo = inner_i + inner_j[::-1]
            elif i == inner_i[0] and j == inner_j[0]:
                combo = inner_i[::-1] + inner_j
            elif i == inner_i[0] and j == inner_j[-1]:
                combo = inner_i[::-1] + inner_j[::-1]
            if combo is None:
                continue
            new_route = [0] + combo + [0]
            # Remove old routes and add new
            if ri < rj:
                routes.pop(rj)
                routes.pop(ri)
            else:
                routes.pop(ri)
                routes.pop(rj)
            routes.append(new_route)
            # Rebuild mappings
            customer_route.clear()
            route_ends.clear()
            for idx, route in enumerate(routes):
                for c in route[1:-1]:
                    customer_route[c] = idx
                if len(route) > 2:
                    route_ends[idx] = (route[1], route[-2])
                else:
                    route_ends[idx] = ()

    def route_distance(route):
        d = 0
        for a, b in zip(route, route[1:]):
            d += distance_matrix[a][b]
        return d

    # Evaluate initial routes
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    # Improvement: relocate from longest to shortest
    max_iter = num_customers * truck_count * 2
    for _ in range(max_iter):
        current_distances = [route_distance(r) for r in best_routes]
        max_idx = int(np.argmax(current_distances))
        min_idx = int(np.argmin(current_distances))
        if current_distances[max_idx] == current_distances[min_idx]:
            break
        longest = best_routes[max_idx]
        shortest = best_routes[min_idx]
        best_new_max = best_max
        best_move = None
        for i in range(1, len(longest)-1):
            cust = longest[i]
            # Build new longest without cust
            new_long = longest[:i] + longest[i+1:]
            if len(new_long) == 2:
                new_long = [0, 0]
            dist_long = route_distance(new_long)
            # Try insert into shortest at each position
            best_short_dist = float('inf')
            best_ins = -1
            for pos in range(1, len(shortest)):
                new_short = shortest[:pos] + [cust] + shortest[pos:]
                d_short = route_distance(new_short)
                if d_short < best_short_dist:
                    best_short_dist = d_short
                    best_ins = pos
            # Compute new max distance
            other_dists = [current_distances[k] for k in range(len(current_distances)) if k != max_idx and k != min_idx]
            new_max = max([dist_long, best_short_dist] + other_dists)
            if new_max < best_new_max - 1e-12:
                best_new_max = new_max
                best_move = (max_idx, min_idx, i, best_ins, cust, new_long, best_short_dist)
        if best_move is not None:
            max_idx, min_idx, i_remove, ins_pos, cust, new_long, _ = best_move
            new_routes = [list(r) for r in best_routes]
            new_routes[max_idx] = new_long
            new_routes[min_idx] = new_routes[min_idx][:ins_pos] + [cust] + new_routes[min_idx][ins_pos:]
            best_routes = new_routes
            best_max = best_new_max
            report_best_vrp(best_routes)
        else:
            break

    # Final sanitization
    final_routes = []
    for r in best_routes:
        if len(r) < 2:
            r = [0, 0]
        else:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
        final_routes.append(r)
    return final_routes