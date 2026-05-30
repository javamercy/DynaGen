import numpy as np

def two_opt(tour, dist):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                a, b = tour[i], tour[i+1]
                c, d = tour[j], tour[(j+1)%n]
                if dist[a,c] + dist[b,d] < dist[a,b] + dist[c,d] - 1e-10:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    break
            if improved:
                break
    cost = dist[tour[-1], tour[0]] + np.sum(dist[tour[:-1], tour[1:]])
    return tour, cost

def double_bridge(tour, rng):
    n = len(tour)
    if n < 4:
        return tour
    i = rng.integers(1, max(2, n//3))
    j = rng.integers(i+1, min(i + n//3, n-2))
    k = rng.integers(j+1, min(j + n//3, n-1))
    A = tour[:i]
    B = tour[i:j]
    C = tour[j:k]
    D = tour[k:]
    new_tour = np.concatenate([A, C, B, D])
    return new_tour

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0,1])
    rng = np.random.default_rng()
    # nearest neighbor construction
    tour = [0]
    unvisited = set(range(1,n))
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda c: distance_matrix[cur,c])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    tour = np.array(tour, dtype=int)
    best_cost = distance_matrix[tour[-1], tour[0]] + np.sum(distance_matrix[tour[:-1], tour[1:]])
    best_tour = tour.copy()
    report_best_tour(best_tour)
    # improve with 2-opt
    tour, cost = two_opt(tour.copy(), distance_matrix)
    if cost < best_cost - 1e-10:
        best_cost = cost
        best_tour = tour.copy()
        report_best_tour(best_tour)
    # iterated local search with perturbation
    for _ in range(20):
        perturbed = double_bridge(tour.copy(), rng)
        improved_tour, improved_cost = two_opt(perturbed, distance_matrix)
        if improved_cost < best_cost - 1e-10:
            best_cost = improved_cost
            best_tour = improved_tour.copy()
            report_best_tour(best_tour)
        tour = improved_tour
    return best_tour