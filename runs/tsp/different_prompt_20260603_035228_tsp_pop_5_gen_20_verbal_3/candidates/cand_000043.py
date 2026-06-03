import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        return np.arange(n, dtype=int)
    
    def nn_construct(start):
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[current, x])
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return np.array(tour, dtype=int)
    
    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    a = tour[i]
                    b = tour[(i + 1) % n]
                    c = tour[j]
                    d = tour[(j + 1) % n]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-9:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True
        return tour
    
    best_tour = nn_construct(0)
    best_dist = np.sum(distance_matrix[best_tour, np.roll(best_tour, -1)])
    report_best_tour(best_tour)
    best_tour = two_opt(best_tour.copy())
    best_dist = np.sum(distance_matrix[best_tour, np.roll(best_tour, -1)])
    report_best_tour(best_tour)
    
    for _ in range(10):
        # random double-bridge perturbation
        tour = best_tour.copy()
        idx = sorted(np.random.choice(n, 4, replace=False))
        segments = [tour[idx[0]:idx[1]], tour[idx[1]:idx[2]], tour[idx[2]:idx[3]], np.concatenate([tour[idx[3]:], tour[:idx[0]]])]
        reordered = np.concatenate([segments[0], segments[2], segments[1], segments[3]])
        # ensure reordered length matches
        if len(reordered) != n:
            continue
        tour = two_opt(reordered)
        dist = np.sum(distance_matrix[tour, np.roll(tour, -1)])
        if dist < best_dist - 1e-9:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    
    return best_tour