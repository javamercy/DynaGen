import numpy as np

def solve_vrp(distance_matrix: np.ndarray, truck_count: int) -> list[list[int]]:
    n = len(distance_matrix)
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)] + [[0, 0]] * (truck_count - (n - 1))
        return routes
    customers = list(range(1, n))
    k = truck_count
    # deterministic farthest-point initialization
    centers = []
    first = max(customers, key=lambda c: distance_matrix[0][c])
    centers.append(first)
    for _ in range(1, k):
        farthest = max(customers, key=lambda c: min(distance_matrix[c][cen] for cen in centers))
        centers.append(farthest)
    # iterative assignment and medoid update
    for _ in range(10):
        clusters = [[] for __ in range(k)]
        for c in customers:
            dists = [distance_matrix[c][cen] for cen in centers]
            cl = min(range(k), key=lambda i: (dists[i], i))
            clusters[cl].append(c)
        # handle empty clusters
        for i in range(k):
            if not clusters[i]:
                largest = max(range(k), key=lambda j: (len(clusters[j]), j))
                if clusters[largest]:
                    moved = clusters[largest].pop()
                    clusters[i].append(moved)
        # compute medoids
        new_centers = []
        for i in range(k):
            if clusters[i]:
                best = clusters[i][0]
                best_sum = sum(distance_matrix[best][oth] for oth in clusters[i])
                for cand in clusters[i][1:]:
                    s = sum(distance_matrix[cand][oth] for oth in clusters[i])
                    if s < best_sum or (s == best_sum and cand < best):
                        best = cand
                        best_sum = s
                new_centers.append(best)
            else:
                new_centers.append(centers[i])
        if set(new_centers) == set(centers):
            break
        centers = new_centers
    # build routes with nearest neighbor
    routes = []
    for cluster in clusters:
        if not cluster:
            routes.append([0, 0])
            continue
        route = [0]
        unvisited = set(cluster)
        while unvisited:
            last = route[-1]
            nearest = min(unvisited, key=lambda c: (distance_matrix[last][c], c))
            route.append(nearest)
            unvisited.remove(nearest)
        route.append(0)
        routes.append(route)
    # 2-opt improvement per route
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
    # helper function for route distance
    def route_dist(route):
        return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    # report initial best
    max_dist = max(route_dist(r) for r in routes)
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)
    # improvement: move customers from longest route
    for _ in range(n * 2):
        longest_idx = max(range(len(routes)), key=lambda i: (route_dist(routes[i]), i))
        longest = routes[longest_idx]
        candidates = sorted([c for c in longest if c != 0])
        best_new_max = float('inf')
        best_move = None
        for cust in candidates:
            new_longest = [x for x in longest if x != cust]
            if len(new_longest) < 2:
                continue
            new_longest_dist = route_dist(new_longest)
            for other_idx in range(len(routes)):
                if other_idx == longest_idx:
                    continue
                other = routes[other_idx]
                for pos in range(1, len(other)):
                    new_other = other[:pos] + [cust] + other[pos:]
                    new_other_dist = route_dist(new_other)
                    other_dists = [route_dist(routes[i]) for i in range(len(routes)) if i not in (longest_idx, other_idx)]
                    cand_max = max(new_longest_dist, new_other_dist, *other_dists)
                    if cand_max < best_new_max:
                        best_new_max = cand_max
                        best_move = (cust, other_idx, pos, new_longest, new_other)
        if best_move is not None and best_new_max < max_dist:
            cust, other_idx, pos, new_longest, new_other = best_move
            routes[longest_idx] = new_longest
            routes[other_idx] = new_other
            max_dist = best_new_max
            best_routes = [list(r) for r in routes]
            report_best_vrp(best_routes)
        else:
            break
    return best_routes