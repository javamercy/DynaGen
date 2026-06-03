import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # Farthest insertion construction
    i, j = np.unravel_index(np.argmax(distance_matrix), (n, n))
    tour = [i, j]
    remaining = set(range(n)) - {i, j}
    while remaining:
        # Find farthest point from tour
        dists = [min(distance_matrix[c, t] for t in tour) for c in remaining]
        farthest = list(remaining)[np.argmax(dists)]
        # Find best insertion position
        best_inc = np.inf
        best_pos = 0
        for k in range(len(tour)):
            a = tour[k]
            b = tour[(k+1) % len(tour)]
            inc = distance_matrix[a, farthest] + distance_matrix[farthest, b] - distance_matrix[a, b]
            if inc < best_inc:
                best_inc = inc
                best_pos = k+1
        tour.insert(best_pos, farthest)
        remaining.remove(farthest)
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    # 2-opt improvement
    best_tour = tour_arr.copy()
    best_cost = sum(distance_matrix[best_tour[i], best_tour[(i+1)%n]] for i in range(n))
    improved = True
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+1, n):
                if (j - i) < 2:
                    continue
                a = best_tour[i]
                b = best_tour[(i+1)%n]
                c = best_tour[j]
                d = best_tour[(j+1)%n]
                delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                if delta < -1e-10:
                    new_tour = best_tour.copy()
                    new_tour[i+1:j+1] = best_tour[j:i:-1]  # reverse segment
                    best_tour = new_tour
                    best_cost += delta
                    improved = True
                    report_best_tour(best_tour)
    return best_tour