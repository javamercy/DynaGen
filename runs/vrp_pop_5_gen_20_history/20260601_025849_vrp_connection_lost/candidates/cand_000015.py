import numpy as np
import heapq

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    num_customers = n - 1
    # Initialize each customer as its own route
    routes = [[0, i, 0] for i in range(1, n)]
    # Add empty routes if more trucks than customers
    for _ in range(truck_count - num_customers):
        routes.append([0, 0])

    if num_customers >= truck_count:
        # Savings list
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                # Negate for max-heap; tie-break by i, then j
                savings.append((-s, i, j))
        heapq.heapify(savings)

        # Track which route each customer belongs to
        customer_route = {i: idx for idx, route in enumerate(routes) for i in route[1:-1]}
        # Track ends of each route (first and last customer)
        route_ends = {}
        for idx, route in enumerate(routes):
            if len(route) > 2:
                route_ends[idx] = [route[1], route[-2]]
            else:
                route_ends[idx] = []

        while len(routes) > truck_count and savings:
            neg_s, i, j = heapq.heappop(savings)
            if i not in customer_route or j not in customer_route:
                continue
            ri = customer_route[i]
            rj = customer_route[j]
            if ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            ends_i = route_ends[ri]
            ends_j = route_ends[rj]
            if i not in ends_i or j not in ends_j:
                continue
            inner_i = route_i[1:-1]
            inner_j = route_j[1:-1]
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
            # Remove the two old routes (order matters for indexing)
            if ri < rj:
                routes.pop(rj)
                routes.pop(ri)
            else:
                routes.pop(ri)
                routes.pop(rj)
            routes.append(new_route)
            # Rebuild customer_route and route_ends
            customer_route.clear()
            route_ends.clear()
            for idx, route in enumerate(routes):
                for c in route[1:-1]:
                    customer_route[c] = idx
                if len(route) > 2:
                    route_ends[idx] = [route[1], route[-2]]
                else:
                    route_ends[idx] = []

    def route_distance(route):
        d = 0
        for a, b in zip(route, route[1:]):
            d += distance_matrix[a][b]
        return d

    # Initial best
    best_routes = [list(r) for r in routes]
    best_max = max(route_distance(r) for r in best_routes)
    report_best_vrp(best_routes)

    max_iter = num_customers * truck_count * 2
    stagnation_limit = max(10, num_customers // 10)
    no_improve = 0

    for _ in range(max_iter):
        improved = False
        # Find longest and shortest route
        dists = [route_distance(r) for r in best_routes]
        max_idx = max(range(len(dists)), key=lambda i: dists[i])
        min_idx = min(range(len(dists)), key=lambda i: dists[i])
        if dists[max_idx] == dists[min_idx]:
            break
        best_delta = 0
        best_move = None
        longest_route = best_routes[max_idx]
        shortest_route = best_routes[min_idx]
        for pos in range(1, len(longest_route)-1):
            cust = longest_route[pos]
            new_long = longest_route[:pos] + longest_route[pos+1:]
            if len(new_long) == 2:
                new_long = [0, 0]
            # Try inserting into shortest route at each position
            for ins in range(1, len(shortest_route)):
                new_short = shortest_route[:ins] + [cust] + shortest_route[ins:]
                d_long = route_distance(new_long)
                d_short = route_distance(new_short)
                # Recompute max distance
                new_max = max(d_long, d_short, max(dists[k] for k in range(len(dists)) if k not in (max_idx, min_idx)))
                if new_max < best_max:
                    delta = best_max - new_max
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (max_idx, min_idx, pos, ins, new_long, new_short)
        if best_move:
            max_idx, min_idx, pos, ins, new_long, new_short = best_move
            new_routes = [list(r) for r in best_routes]
            new_routes[max_idx] = new_long
            new_routes[min_idx] = new_short
            new_max = max(route_distance(r) for r in new_routes)
            if new_max < best_max:
                best_max = new_max
                best_routes = new_routes
                report_best_vrp(best_routes)
                improved = True
                no_improve = 0
        if not improved:
            no_improve += 1
            if no_improve >= stagnation_limit:
                break

    # Ensure proper formatting
    final_routes = []
    for r in best_routes:
        if len(r) == 2:
            r = [0, 0]
        else:
            if r[0] != 0:
                r.insert(0, 0)
            if r[-1] != 0:
                r.append(0)
        final_routes.append(r)
    return final_routes