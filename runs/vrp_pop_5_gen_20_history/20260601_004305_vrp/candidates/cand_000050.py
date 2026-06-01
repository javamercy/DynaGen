import numpy as np

def solve_vrp(distance_matrix, truck_count):
    n = len(distance_matrix)
    if truck_count >= n - 1:
        routes = [[0, i, 0] for i in range(1, n)] + [[0, 0]] * (truck_count - (n - 1))
        report_best_vrp(routes)
        return routes
    customers = list(range(1, n))
    k = truck_count
    # farthest-point initialization
    seeds = []
    first = max(customers, key=lambda c: distance_matrix[0][c])
    seeds.append(first)
    for _ in range(1, k):
        farthest = max(customers, key=lambda c: min(distance_matrix[c][seed] for seed in seeds))
        seeds.append(farthest)
    # assign customers to nearest seed (deterministic tie-break by seed index)
    clusters = [[] for _ in range(k)]
    for c in customers:
        dists = [distance_matrix[c][seed] for seed in seeds]
        cl = min(range(k), key=lambda i: (dists[i], i))
        clusters[cl].append(c)
    # handle empty clusters: move from largest
    for i in range(k):
        if not clusters[i]:
            largest = max(range(k), key=lambda j: (len(clusters[j]), j))
            if clusters[largest]:
                moved = clusters[largest].pop()
                clusters[i].append(moved)
    # build routes with nearest neighbor (starting from depot, ends at depot)
    routes = []
    for cluster in clusters:
        if not cluster:
            routes.append([0, 0])
            continue
        route = [0]
        unvisited = set(cluster)
        while unvisited:
            last = route[-1]
            nxt = min(unvisited, key=lambda c: (distance_matrix[last][c], c))
            route.append(nxt)
            unvisited.remove(nxt)
        route.append(0)
        routes.append(route)
    # 2-opt improvement per route
    def route_dist(r):
        return sum(distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1))
    for idx in range(len(routes)):
        route = routes[idx]
        improved = True
        max_iter = len(route) * len(route)
        for _ in range(max_iter):
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
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
    # compute initial max distance
    max_dist = max(route_dist(r) for r in routes)
    best_routes = [list(r) for r in routes]
    report_best_vrp(best_routes)
    # first-improvement relocate from longest route
    for _ in range(n * 2):
        cur_max = max(route_dist(r) for r in routes)
        longest_idx = max(range(len(routes)), key=lambda i: (route_dist(routes[i]), i))
        longest = routes[longest_idx]
        found = False
        for idx_c in range(1, len(longest)-1):
            cust = longest[idx_c]
            new_longest = longest[:idx_c] + longest[idx_c+1:]
            dist_longest = route_dist(new_longest)
            for other_idx in range(len(routes)):
                if other_idx == longest_idx:
                    continue
                other = routes[other_idx]
                for pos in range(1, len(other)):
                    new_other = other[:pos] + [cust] + other[pos:]
                    dist_other = route_dist(new_other)
                    new_max = dist_longest if dist_longest > dist_other else dist_other
                    for ri in range(len(routes)):
                        if ri != longest_idx and ri != other_idx:
                            d = route_dist(routes[ri])
                            if d > new_max:
                                new_max = d
                    if new_max < cur_max:
                        # apply move
                        routes[longest_idx] = new_longest
                        routes[other_idx] = new_other
                        best_routes = [list(r) for r in routes]
                        report_best_vrp(best_routes)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return best_routes