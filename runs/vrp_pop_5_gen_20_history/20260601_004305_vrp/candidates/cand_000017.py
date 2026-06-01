import numpy as np
import math

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = distance_matrix.shape[0]
    customers = list(range(1, n))
    m = n - 1
    if truck_count >= m:
        routes = [[0, i, 0] for i in customers] + [[0, 0]] * (truck_count - m)
        report_best_vrp(routes)
        return routes

    # TSP tour using nearest neighbor, starting from depot with smallest index tie-break
    tour = []
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(m):
        best = -1
        best_dist = math.inf
        for v in range(1, n):
            if not visited[v]:
                d = distance_matrix[current][v]
                if d < best_dist or (d == best_dist and v < best):
                    best_dist = d
                    best = v
        tour.append(best)
        visited[best] = True
        current = best

    custs = tour
    k = truck_count
    # Precompute segment distances
    seg_dist = [[0] * (m + 1) for _ in range(m)]
    for l in range(m):
        acc = distance_matrix[0][custs[l]]
        for r in range(l + 1, m + 1):
            if r > l + 1:
                acc += distance_matrix[custs[r - 2]][custs[r - 1]]
            if r == l + 1:
                route_dist = distance_matrix[0][custs[l]] + distance_matrix[custs[l]][0]
            else:
                route_dist = acc + distance_matrix[custs[r - 1]][0]
            seg_dist[l][r] = route_dist

    # DP: dp[i][t] = min max distance for first i customers with t routes
    dp = [[math.inf] * (k + 1) for _ in range(m + 1)]
    choice = [[-1] * (k + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    for i in range(1, m + 1):
        for t in range(1, min(i, k) + 1):
            best_val = math.inf
            best_j = -1
            for j in range(t - 1, i):
                if dp[j][t - 1] < math.inf:
                    cand = max(dp[j][t - 1], seg_dist[j][i])
                    if cand < best_val or (cand == best_val and j < best_j):
                        best_val = cand
                        best_j = j
            dp[i][t] = best_val
            choice[i][t] = best_j

    # Reconstruct routes
    routes = []
    i = m
    t = k
    while t > 0:
        j = choice[i][t]
        seg = custs[j:i]
        routes.append([0] + seg + [0])
        i = j
        t -= 1
    routes.reverse()
    while len(routes) < truck_count:
        routes.append([0, 0])

    # Helper functions
    def route_dist(route):
        return sum(distance_matrix[route[a]][route[a+1]] for a in range(len(route)-1))

    def compute_max():
        return max(route_dist(r) for r in routes)

    current_max = compute_max()
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)

    # Local search: best-improvement relocate and swap
    max_iter = 100 * n
    for _ in range(max_iter):
        best_improvement = 0.0
        best_move = None  # (type, cust_i, cust_j, route_i, route_j, pos_i, pos_j)
        # Relocate moves
        for cust in range(1, n):
            # Find route and position of cust
            ri = None
            posi = None
            for idx, route in enumerate(routes):
                if cust in route:
                    ri = idx
                    posi = route.index(cust)
                    break
            if ri is None:
                continue
            # Try inserting into other routes
            for rj in range(len(routes)):
                if rj == ri:
                    continue
                other = routes[rj]
                for pos in range(1, len(other)):
                    # Evaluate move: remove cust from ri, insert at pos in rj
                    new_ri = routes[ri][:posi] + routes[ri][posi+1:]
                    new_rj = other[:pos] + [cust] + other[pos:]
                    # Compute new max quickly
                    new_max = max(route_dist(new_ri), route_dist(new_rj))
                    for rr in range(len(routes)):
                        if rr != ri and rr != rj:
                            d = route_dist(routes[rr])
                            if d > new_max:
                                new_max = d
                    improvement = current_max - new_max
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = ('relocate', cust, None, ri, rj, posi, pos)
                    elif improvement == best_improvement and improvement > 0:
                        # Tie-break by smallest customer, then smallest destination route, then position
                        if best_move is None:
                            best_move = ('relocate', cust, None, ri, rj, posi, pos)
                        else:
                            # Compare tie-breaking rules
                            if cust < best_move[1] or (cust == best_move[1] and rj < best_move[4]):
                                best_move = ('relocate', cust, None, ri, rj, posi, pos)
        # Swap moves
        for i in range(1, n):
            ri = None
            posi = None
            for idx, route in enumerate(routes):
                if i in route:
                    ri = idx
                    posi = route.index(i)
                    break
            if ri is None:
                continue
            for j in range(i+1, n):
                rj = None
                posj = None
                for idx, route in enumerate(routes):
                    if j in route:
                        rj = idx
                        posj = route.index(j)
                        break
                if rj is None or ri == rj:
                    continue
                # Evaluate swap i and j
                # Start with copies
                new_ri = list(routes[ri])
                new_rj = list(routes[rj])
                # Remove i from ri, j from rj
                new_ri.pop(posi)
                new_rj.pop(posj)
                # Insert i into rj at original position of j (after removal, posj is still valid if we adjust? Actually after removal, the list length decreased; we want to insert at the same index relative to the original? Better: insert i at posj in new_rj, and j at posi in new_ri.
                new_ri.insert(posi, j)
                new_rj.insert(posj, i)
                new_max = max(route_dist(new_ri), route_dist(new_rj))
                for rr in range(len(routes)):
                    if rr != ri and rr != rj:
                        d = route_dist(routes[rr])
                        if d > new_max:
                            new_max = d
                improvement = current_max - new_max
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = ('swap', i, j, ri, rj, posi, posj)
                elif improvement == best_improvement and improvement > 0:
                    if best_move is None:
                        best_move = ('swap', i, j, ri, rj, posi, posj)
                    else:
                        if i < best_move[1] or (i == best_move[1] and j < best_move[2]):
                            best_move = ('swap', i, j, ri, rj, posi, posj)
        if best_move and best_improvement > 0:
            # Apply move
            if best_move[0] == 'relocate':
                _, cust, _, ri, rj, posi, pos = best_move
                routes[ri].pop(posi)
                routes[rj].insert(pos, cust)
            else:
                _, i, j, ri, rj, posi, posj = best_move
                routes[ri][posi] = j
                routes[rj][posj] = i
            current_max = compute_max()
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        else:
            break

    # Intra-route 2-opt improvement on each route
    for idx in range(len(routes)):
        route = routes[idx]
        for _ in range(1000):
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    if j - i == 1:
                        continue
                    a, b, c, d = route[i-1], route[i], route[j], route[j+1]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new = distance_matrix[a][c] + distance_matrix[b][d]
                    if new < old:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        routes[idx] = route
    # Update best after 2-opt
    new_max = compute_max()
    if new_max < current_max:
        best_routes = [list(r) for r in routes]
        report_best_vrp(best_routes)
    return best_routes