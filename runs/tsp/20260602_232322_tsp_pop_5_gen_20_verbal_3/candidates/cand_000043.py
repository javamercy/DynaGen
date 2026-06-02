import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n)
    # random initialization
    tour = np.random.permutation(n)
    best_tour = tour.copy()
    best_dist = sum(distance_matrix[tour[i-1], tour[i]] for i in range(n))
    report_best_tour(best_tour)
    
    # simulated annealing
    T = 1.0
    T_min = 1e-4
    alpha = 0.995
    while T > T_min:
        i, j = np.random.choice(n, 2, replace=False)
        if i > j:
            i, j = j, i
        # ensure non-adjacent edges (i,i+1) and (j,j+1)
        if (j - i) % n == 1 or (i - j) % n == 1:
            continue
        a, b = tour[i], tour[(i+1)%n]
        c, d = tour[j], tour[(j+1)%n]
        delta = distance_matrix[a,c] + distance_matrix[b,d] - (distance_matrix[a,b] + distance_matrix[c,d])
        if delta < 0 or np.random.rand() < np.exp(-delta/T):
            # apply 2-opt: reverse segment (i+1 .. j)
            segment = tour[i+1:j+1]
            tour[i+1:j+1] = segment[::-1]
            new_dist = sum(distance_matrix[tour[k-1], tour[k]] for k in range(n))
            if new_dist < best_dist:
                best_dist = new_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
        T *= alpha
    return best_tour