import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n, dtype=np.int32)
        report_best_tour(tour)
        return tour
    # nearest neighbor construction
    tour = [0]
    unvisited = np.ones(n, dtype=bool)
    unvisited[0] = False
    current = 0
    for _ in range(1, n):
        dists = distance_matrix[current, unvisited]
        next_city = np.where(unvisited)[0][np.argmin(dists)]
        tour.append(int(next_city))
        unvisited[next_city] = False
        current = next_city
    tour = np.array(tour, dtype=np.int32)
    report_best_tour(tour)
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        ext = np.empty(n+1, dtype=np.int32)
        ext[:n] = tour
        ext[n] = tour[0]
        for i in range(n-1):
            for j in range(i+2, n):
                delta = (distance_matrix[ext[i], ext[j]] +
                         distance_matrix[ext[i+1], ext[j+1]] -
                         distance_matrix[ext[i], ext[i+1]] -
                         distance_matrix[ext[j], ext[j+1]])
                if delta < -1e-12:
                    ext[i+1:j+1] = ext[i+1:j+1][::-1]
                    improved = True
                    tour = ext[:n].copy()
                    report_best_tour(tour)
                    break
            if improved:
                break
    # node insertion improvement
    for _ in range(5):
        improved = False
        for u in range(n):
            node = tour[u]
            new_tour = np.delete(tour, u)
            best_delta = 0
            best_pos = 0
            # consider inserting before each position (including at end)
            for pos in range(n):
                prev = new_tour[(pos-1) % n]
                next_city = new_tour[pos % n]
                delta = (distance_matrix[prev, node] +
                         distance_matrix[node, next_city] -
                         distance_matrix[prev, next_city])
                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_pos = pos
            if best_delta < -1e-12:
                new_tour = np.insert(new_tour, best_pos, node)
                tour = new_tour
                improved = True
                report_best_tour(tour)
        if not improved:
            break
    return tour