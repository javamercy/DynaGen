import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # Nearest neighbor construction starting from 0
    tour = [0]
    visited = {0}
    current = 0
    for _ in range(n - 1):
        next_node = None
        min_dist = np.inf
        for k in range(n):
            if k not in visited and distance_matrix[current, k] < min_dist:
                min_dist = distance_matrix[current, k]
                next_node = k
        tour.append(next_node)
        visited.add(next_node)
        current = next_node
    tour = np.array(tour, dtype=np.int64)
    
    def tour_length(t):
        return distance_matrix[t[-1], t[0]] + np.sum(distance_matrix[t[i], t[i+1]] for i in range(n-1))
    
    best_tour = tour.copy()
    best_len = tour_length(best_tour)
    report_best_tour(best_tour)
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 3):
            for j in range(i + 2, n - 1):
                # current edges: (i,i+1) and (j,j+1), new edges: (i,j) and (i+1,j+1)
                a = tour[i]
                b = tour[i+1]
                c = tour[j]
                d = tour[(j+1) % n]
                old = distance_matrix[a, b] + distance_matrix[c, d]
                new = distance_matrix[a, c] + distance_matrix[b, d]
                if new < old:
                    # reverse segment from i+1 to j
                    tour = np.concatenate((tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]))
                    improved = True
                    current_len = best_len - old + new
                    if current_len < best_len:
                        best_len = current_len
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour