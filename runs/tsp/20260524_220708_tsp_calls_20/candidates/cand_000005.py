import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(distance_matrix)
    # Build initial tour via randomized greedy with RCL
    unvisited = set(range(n))
    start = rng.integers(n)
    tour = [start]
    unvisited.remove(start)
    k = max(3, int(np.sqrt(n)))
    current = start
    while unvisited:
        # compute distances to unvisited
        dists = distance_matrix[current, list(unvisited)]
        # sort indices of unvisited by distance
        sorted_indices = sorted(unvisited, key=lambda x: distance_matrix[current, x])
        # RCL: pick among first k
        rcl_size = min(k, len(sorted_indices))
        candidate = sorted_indices[rng.integers(rcl_size)]
        tour.append(candidate)
        unvisited.remove(candidate)
        current = candidate
    tour = np.array(tour, dtype=int)
    # compute initial distance
    def tour_length(t):
        return distance_matrix[t[-1], t[0]] + np.sum(distance_matrix[t[:-1], t[1:]])
    best_tour = tour.copy()
    best_len = tour_length(tour)
    report_best_tour(best_tour)
    # 2-opt improvement
    improved = True
    while budget > 0 and improved:
        improved = False
        for i in range(n - 2):
            if budget <= 0:
                break
            for j in range(i + 2, n):
                if budget <= 0:
                    break
                # consider swapping edges (i, i+1) and (j, j+1)
                # new edges: (i, j) and (i+1, j+1)
                # calculate delta
                a, b = tour[i], tour[i+1]
                c, d = tour[j], tour[(j+1)%n]
                old = distance_matrix[a,b] + distance_matrix[c,d]
                new = distance_matrix[a,c] + distance_matrix[b,d]
                if new < old:
                    # perform swap
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    new_len = best_len - old + new
                    if new_len < best_len:
                        best_len = new_len
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                budget -= 1
        # one full pass counts as budget usage; each inner iteration counted already
    return best_tour