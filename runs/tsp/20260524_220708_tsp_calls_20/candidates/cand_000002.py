import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        return tour
    np.random.seed(seed)
    perm = np.random.permutation(n)
    tour = [perm[0], perm[1]]
    unvisited = set(perm[2:])
    while unvisited:
        best_inc = float('inf')
        candidates = []
        for city in unvisited:
            for i in range(len(tour)):
                prev = tour[i]
                nxt = tour[(i+1) % len(tour)]
                inc = distance_matrix[prev, city] + distance_matrix[city, nxt] - distance_matrix[prev, nxt]
                if inc < best_inc:
                    best_inc = inc
                    candidates = [(city, i)]
                elif inc == best_inc:
                    candidates.append((city, i))
        idx = np.random.randint(len(candidates))
        city, pos = candidates[idx]
        tour.insert(pos+1, city)
        unvisited.remove(city)
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    tour = tour_arr.tolist()
    effort = 0
    improved = True
    while effort < budget and improved:
        improved = False
        best_delta = 0
        best_i = best_j = -1
        for i in range(n):
            for j in range(i+2, n):
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        if best_delta < 0:
            tour[best_i+1:best_j+1] = reversed(tour[best_i+1:best_j+1])
            improved = True
            effort += 1
            report_best_tour(np.array(tour))
    return np.array(tour)