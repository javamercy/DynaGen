import numpy as np

def solve_tsp(dm):
    n = len(dm)
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour
    
    # nearest neighbor tour starting from 0
    tour = [0]
    unvisited = set(range(1, n))
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda x: dm[cur, x])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    tour = np.array(tour, dtype=np.int32)
    report_best_tour(tour)
    best_cost = tour_cost(tour, dm)
    
    # 2-opt improvement
    tour = two_opt(tour, dm, best_cost)
    report_best_tour(tour)
    
    # divide and conquer recursive improvement
    tour = divide_conquer(dm, np.arange(n), 3)
    report_best_tour(tour)
    return tour

def tour_cost(tour, dm):
    n = len(tour)
    cost = dm[tour[-1], tour[0]]
    for i in range(n-1):
        cost += dm[tour[i], tour[i+1]]
    return cost

def two_opt(tour, dm, best_cost):
    improved = True
    n = len(tour)
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                if j == n-1 and i == 0:
                    continue
                delta = (dm[tour[i], tour[j]] + dm[tour[i+1], tour[(j+1)%n]] -
                         dm[tour[i], tour[i+1]] - dm[tour[j], tour[(j+1)%n]])
                if delta < -1e-10:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    best_cost += delta
                    improved = True
        if improved:
            report_best_tour(tour)
    return tour

def divide_conquer(dm, indices, depth):
    if len(indices) <= 20 or depth == 0:
        # solve small subproblem with NN + 2-opt
        tour = nearest_neighbor_subset(dm, indices)
        return two_opt(tour, dm, tour_cost(tour, dm))
    # find two farthest nodes
    max_dist = -1
    seed1 = seed2 = indices[0]
    for i in indices:
        for j in indices:
            if i < j and dm[i, j] > max_dist:
                max_dist = dm[i, j]
                seed1, seed2 = i, j
    clusters = [[] for _ in range(2)]
    for idx in indices:
        dist1 = dm[idx, seed1]
        dist2 = dm[idx, seed2]
        if dist1 < dist2:
            clusters[0].append(idx)
        else:
            clusters[1].append(idx)
    # ensure each cluster has at least 2 nodes
    if len(clusters[0]) < 2:
        clusters[0].append(clusters[1].pop())
    if len(clusters[1]) < 2:
        clusters[1].append(clusters[0].pop())
    # recursively solve
    tour1 = divide_conquer(dm, np.array(clusters[0], dtype=np.int32), depth-1)
    tour2 = divide_conquer(dm, np.array(clusters[1], dtype=np.int32), depth-1)
    # merge tours
    n1, n2 = len(tour1), len(tour2)
    best_cost = np.inf
    best_merge = None
    for i in range(n1):
        for j in range(n2):
            for rev1 in [False, True]:
                for rev2 in [False, True]:
                    if rev1:
                        t1 = np.roll(tour1[::-1], i)
                    else:
                        t1 = np.roll(tour1, i)
                    if rev2:
                        t2 = np.roll(tour2[::-1], j)
                    else:
                        t2 = np.roll(tour2, j)
                    cand = np.concatenate([t1, t2])
                    cost = tour_cost(cand, dm)
                    if cost < best_cost:
                        best_cost = cost
                        best_merge = cand
    # improve merged tour with 2-opt
    merged = two_opt(best_merge, dm, best_cost)
    return merged

def nearest_neighbor_subset(dm, indices):
    if len(indices) == 1:
        return indices.astype(np.int32)
    start = indices[0]
    tour = [start]
    unvisited = set(indices[1:])
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda x: dm[cur, x])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return np.array(tour, dtype=np.int32)