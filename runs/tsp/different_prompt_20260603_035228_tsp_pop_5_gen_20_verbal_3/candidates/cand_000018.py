import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 1:
        return np.arange(n).astype(int)
    best_tour = None
    best_dist = np.inf
    for _ in range(10):
        start = np.random.randint(n)
        tour = [start]
        unvisited = set(range(n)) - {start}
        current = start
        while unvisited:
            dists = [distance_matrix[current, node] for node in unvisited]
            min_dist = min(dists)
            candidates = [node for node, d in zip(unvisited, dists) if abs(d - min_dist) < 1e-12]
            next_node = np.random.choice(candidates)
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        tour = np.array(tour, dtype=int)
        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    a, b = tour[i], tour[(i + 1) % n]
                    c, d = tour[j], tour[(j + 1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-9:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        # update best
        cur_dist = distance_matrix[tour[-1], tour[0]] + sum(distance_matrix[tour[k], tour[k+1]] for k in range(n-1))
        if cur_dist < best_dist - 1e-9:
            best_dist = cur_dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    return best_tour