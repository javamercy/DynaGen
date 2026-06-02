import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    # start with two random nodes
    indices = np.random.permutation(n)
    tour = [indices[0], indices[1]]
    unvisited = set(range(n)) - {tour[0], tour[1]}
    
    while unvisited:
        # compute best insertion cost for each unvisited node
        candidates = []  # (cost, node, position)
        for node in unvisited:
            best_cost = np.inf
            best_pos = 0
            for i in range(len(tour)):
                p = tour[i]
                q = tour[(i + 1) % len(tour)]
                delta = distance_matrix[p, node] + distance_matrix[node, q] - distance_matrix[p, q]
                if delta < best_cost:
                    best_cost = delta
                    best_pos = i + 1
            candidates.append((best_cost, node, best_pos))
        # sort by cost
        candidates.sort(key=lambda x: x[0])
        # choose randomly from top sqrt(|unvisited|)
        k = max(1, int(np.sqrt(len(unvisited))))
        idx = np.random.randint(min(k, len(candidates)))
        _, node, pos = candidates[idx]
        tour.insert(pos, node)
        unvisited.remove(node)
    
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == i + 1:
                    continue
                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[j], tour[(j + 1) % n]
                old = distance_matrix[a, b] + distance_matrix[c, d]
                new = distance_matrix[a, c] + distance_matrix[b, d]
                if new < old:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True
                    tour_arr = np.array(tour)
                    report_best_tour(tour_arr)
                    break
            if improved:
                break
    return tour_arr