import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # nearest neighbor initial tour
    start = 0
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur = start
    while unvisited:
        next_city = min(unvisited, key=lambda x: distance_matrix[cur, x])
        tour.append(next_city)
        unvisited.remove(next_city)
        cur = next_city
    tour = np.array(tour, dtype=int)
    # compute initial distance
    cur_dist = sum(distance_matrix[tour[i], tour[(i+1) % n]] for i in range(n))
    best_tour = tour.copy()
    best_dist = cur_dist
    report_best_tour(best_tour)
    # SA parameters
    T0 = best_dist * 0.1
    T = T0
    T_final = 1e-8
    alpha = 0.999
    max_iter = 10000
    for _ in range(max_iter):
        # random 2-opt move
        i = np.random.randint(0, n - 1)
        j = np.random.randint(i + 1, n)
        if j - i == 1:
            continue
        # compute delta distance
        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n]
        delta = distance_matrix[a, c] + distance_matrix[b, d] - (distance_matrix[a, b] + distance_matrix[c, d])
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # perform reverse
            tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
            cur_dist += delta
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_tour = tour.copy()
                report_best_tour(best_tour)
        # cooling
        T *= alpha
        if T < T_final:
            break
    return best_tour