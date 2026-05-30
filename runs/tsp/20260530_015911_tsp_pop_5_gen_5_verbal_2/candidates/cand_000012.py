import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.array(range(n))
    # Farthest insertion construction
    max_dist = -1
    start_pair = (0, 1)
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i][j] > max_dist:
                max_dist = distance_matrix[i][j]
                start_pair = (i, j)
    tour = [start_pair[0], start_pair[1]]
    unvisited = set(range(n)) - {start_pair[0], start_pair[1]}
    while unvisited:
        best_node = None
        best_increase = -1
        best_pos = None
        for node in unvisited:
            min_increase = float('inf')
            min_pos = None
            m = len(tour)
            for i in range(m):
                j = (i + 1) % m
                inc = distance_matrix[tour[i]][node] + distance_matrix[node][tour[j]] - distance_matrix[tour[i]][tour[j]]
                if inc < min_increase:
                    min_increase = inc
                    min_pos = j
            if min_increase > best_increase:
                best_increase = min_increase
                best_node = node
                best_pos = min_pos
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
    tour = np.array(tour, dtype=int)
    def _tour_length(t):
        total = 0.0
        for k in range(n):
            total += distance_matrix[t[k]][t[(k+1) % n]]
        return total
    best_dist = _tour_length(tour)
    report_best_tour(tour.copy())
    # VND with limited iterations
    max_iter = 5
    for _ in range(max_iter):
        improved = False
        # 2-opt first improvement
        for i in range(n-2):
            for j in range(i+2, n-1):
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1) % n]
                delta = distance_matrix[a][c] + distance_matrix[b][d] - distance_matrix[a][b] - distance_matrix[c][d]
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    new_dist = _tour_length(tour)
                    if new_dist < best_dist - 1e-12:
                        best_dist = new_dist
                        report_best_tour(tour.copy())
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Or-opt single pass, break on first improvement overall
            for i in range(n):
                node = tour[i]
                tour_no_i = np.delete(tour, i)
                best_candidate = None
                best_cand_dist = best_dist
                for j in range(n):  # insertion positions
                    candidate = np.insert(tour_no_i, j, node)
                    cand_dist = _tour_length(candidate)
                    if cand_dist < best_cand_dist - 1e-12:
                        best_cand_dist = cand_dist
                        best_candidate = candidate
                if best_candidate is not None:
                    tour = best_candidate
                    best_dist = best_cand_dist
                    report_best_tour(tour.copy())
                    improved = True
                    break
        if not improved:
            break
    return tour