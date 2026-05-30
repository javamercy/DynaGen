import numpy as np

def two_opt(tour, dist, max_iter=500):
    n = len(tour)
    improved = True
    improvements = 0
    while improved and improvements < max_iter:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                a, b = tour[i], tour[i+1]
                c, d = tour[j], tour[(j+1)%n]
                if dist[a,c] + dist[b,d] < dist[a,b] + dist[c,d] - 1e-10:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    improvements += 1
                    break
            if improved:
                break
    cost = dist[tour[-1], tour[0]] + np.sum(dist[tour[:-1], tour[1:]])
    return tour, cost

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0,1])
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
    # 2-opt with limited improvements
    tour, cost = two_opt(tour, distance_matrix, max_iter=500)
    if cost < best_cost - 1e-10:
        best_cost = cost
        best_tour = tour.copy()
        report_best_tour(best_tour)
    return best_tour