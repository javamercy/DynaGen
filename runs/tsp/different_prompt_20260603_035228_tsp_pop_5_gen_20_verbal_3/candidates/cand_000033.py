import numpy as np
def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n, dtype=int)
    # nearest neighbor
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        next_node = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    tour = np.array(tour, dtype=int)
    report_best_tour(tour)
    best_tour = tour.copy()
    best_len = _tour_length(distance_matrix, tour)
    
    improved = True
    while improved:
        # 2-opt
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                a = tour[i]
                b = tour[(i + 1) % n]
                c = tour[j]
                d = tour[(j + 1) % n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
                    report_best_tour(tour)
                    cur_len = _tour_length(distance_matrix, tour)
                    if cur_len < best_len:
                        best_len = cur_len
                        best_tour = tour.copy()
                    break
            if improved:
                break
        if not improved:
            # perturbation: random segment reversal
            i = np.random.randint(0, n - 1)
            j = np.random.randint(i + 2, n)
            tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
            cur_len = _tour_length(distance_matrix, tour)
            if cur_len < best_len:
                best_len = cur_len
                best_tour = tour.copy()
                report_best_tour(tour)
            improved = True  # attempt 2-opt again
    return best_tour.astype(int)

def _tour_length(dm, tour):
    n = len(tour)
    total = dm[tour[-1], tour[0]]
    for i in range(n-1):
        total += dm[tour[i], tour[i+1]]
    return total