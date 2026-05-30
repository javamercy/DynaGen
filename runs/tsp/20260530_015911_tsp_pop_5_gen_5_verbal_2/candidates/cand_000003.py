import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    # Nearest neighbor initial tour
    tour = np.zeros(n, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    current = 0
    visited[current] = True
    tour[0] = current
    for i in range(1, n):
        dist = distance_matrix[current]
        dist[visited] = np.inf
        current = np.argmin(dist)
        visited[current] = True
        tour[i] = current
    
    best_tour = tour.copy()
    best_dist = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
    report_best_tour(best_tour)
    
    # Simulated annealing parameters
    T_start = np.max(distance_matrix) * n
    T_end = 1e-8
    max_iter = 50000
    cooling = lambda k: T_start * (T_end/T_start) ** (k / max_iter)
    
    tour = best_tour.copy()
    current_dist = best_dist
    
    for iteration in range(max_iter):
        T = cooling(iteration)
        # Generate random 2-opt move
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i > j:
            i, j = j, i
        if j - i < 2:
            continue
        # Compute delta
        i_next = (i + 1) % n
        j_next = (j + 1) % n
        delta = (distance_matrix[tour[i], tour[j]] +
                 distance_matrix[tour[i_next], tour[j_next]] -
                 distance_matrix[tour[i], tour[i_next]] -
                 distance_matrix[tour[j], tour[j_next]])
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Reverse segment i+1..j
            tour[i_next:j+1] = tour[i_next:j+1][::-1]
            current_dist += delta
            if current_dist < best_dist:
                best_dist = current_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
    return best_tour