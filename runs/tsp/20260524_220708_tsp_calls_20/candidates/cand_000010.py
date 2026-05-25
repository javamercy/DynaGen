import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
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
        for i in range(n):
            for j in range(i+2, n):
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < 0:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
                    effort += 1
                    report_best_tour(np.array(tour))
                    break
            if improved:
                break
    return np.array(tour)