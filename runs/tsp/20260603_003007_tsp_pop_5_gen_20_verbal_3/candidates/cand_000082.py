import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    
    # Nearest neighbor construction
    start = np.random.randint(n)
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
    dist = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    
    def two_opt(t, d):
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    if j - i == 1:
                        continue
                    a, b = t[i], t[(i+1)%n]
                    c, d_ = t[j], t[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d_] - distance_matrix[a,b] - distance_matrix[c,d_]
                    if delta < -1e-12:
                        t[i+1:j+1] = t[i+1:j+1][::-1]
                        d += delta
                        improved = True
        return t, d
    
    tour, dist = two_opt(tour, dist)
    best_tour = tour.copy()
    best_dist = dist
    cur_tour = tour.copy()
    cur_dist = dist
    report_best_tour(best_tour)
    
    # Iterated local search with random swaps
    num_perturb = min(20, max(5, n // 10))
    for _ in range(num_perturb):
        i, j = np.random.choice(n, 2, replace=False)
        new_tour = cur_tour.copy()
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_dist = sum(distance_matrix[new_tour[i], new_tour[(i+1)%n]] for i in range(n))
        new_tour, new_dist = two_opt(new_tour, new_dist)
        if new_dist < best_dist:
            best_dist = new_dist
            best_tour = new_tour.copy()
            report_best_tour(best_tour)
            cur_tour = new_tour
            cur_dist = new_dist
        elif new_dist < cur_dist:
            cur_tour = new_tour
            cur_dist = new_dist
    return best_tour