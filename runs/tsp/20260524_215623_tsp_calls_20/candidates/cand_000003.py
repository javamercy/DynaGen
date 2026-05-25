import numpy as np
def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    max_idx = np.unravel_index(np.argmax(distance_matrix), (n, n))
    tour = list(max_idx)
    unvisited = set(range(n)) - set(tour)
    used_budget = 0
    while unvisited and used_budget < budget:
        best_node = None
        best_inc = -1.0
        best_pos = 0
        for node in unvisited:
            min_inc = np.inf
            ins_pos = 0
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                inc = distance_matrix[tour[i], node] + distance_matrix[node, tour[j]] - distance_matrix[tour[i], tour[j]]
                if inc < min_inc - 1e-12:
                    min_inc = inc
                    ins_pos = i + 1
                elif abs(inc - min_inc) < 1e-12 and np.random.rand() < 0.5:
                    ins_pos = i + 1
            if min_inc > best_inc:
                best_inc = min_inc
                best_node = node
                best_pos = ins_pos
            elif abs(min_inc - best_inc) < 1e-12 and np.random.rand() < 0.5:
                best_node = node
                best_pos = ins_pos
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
        used_budget += 1
        report_best_tour(np.array(tour))
    improved = True
    while improved and used_budget < budget:
        improved = False
        for i in range(n):
            for k in range(i + 1, n):
                if k - i == 1:
                    continue
                a = tour[i]
                b = tour[(i + 1) % n]
                c = tour[k]
                d = tour[(k + 1) % n]
                delta = -distance_matrix[a, b] - distance_matrix[c, d] + distance_matrix[a, c] + distance_matrix[b, d]
                if delta < -1e-12:
                    tour[i+1:k+1] = tour[i+1:k+1][::-1]
                    improved = True
                    used_budget += 1
                    report_best_tour(np.array(tour))
                    if used_budget >= budget:
                        break
            if used_budget >= budget:
                break
    return np.array(tour)