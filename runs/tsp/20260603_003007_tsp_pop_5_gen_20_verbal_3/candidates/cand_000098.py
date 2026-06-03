import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # Nearest neighbor
    start = 0
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda x: distance_matrix[cur, x])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    tour = np.array(tour)
    
    def tsp_dist(t):
        return distance_matrix[t[-1], t[0]] + sum(distance_matrix[t[i], t[i+1]] for i in range(n-1))
    
    best_tour = tour.copy()
    best_dist = tsp_dist(tour)
    report_best_tour(best_tour)
    
    def two_opt(t):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    if j - i == 1:
                        continue
                    a, b = t[i], t[(i+1)%n]
                    c, d = t[j], t[(j+1)%n]
                    if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                        t[i+1:j+1] = t[i+1:j+1][::-1]
                        improved = True
        return t
    
    best_tour = two_opt(tour.copy())
    best_dist = tsp_dist(best_tour)
    report_best_tour(best_tour)
    
    max_iter = 200
    stagnation = 0
    max_stagnation = 20
    for _ in range(max_iter):
        # double-bridge perturbation
        i1, i2, i3 = 1 + np.random.randint(0, n-3), 1 + np.random.randint(i1+2, n-1), 1 + np.random.randint(i2+2, n)
        i1, i2, i3 = sorted([i1, i2, i3])
        new_tour = np.empty(n, dtype=int)
        new_tour[:i1] = best_tour[:i1]
        new_tour[i1:i2] = best_tour[i2:i3]
        new_tour[i2:i3] = best_tour[i1:i2]
        new_tour[i3:] = best_tour[i3:]
        new_tour = two_opt(new_tour)
        new_dist = tsp_dist(new_tour)
        if new_dist < best_dist:
            best_tour = new_tour.copy()
            best_dist = new_dist
            report_best_tour(best_tour)
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= max_stagnation:
                # restart from random tour
                perm = np.random.permutation(n)
                new_tour = two_opt(perm)
                new_dist = tsp_dist(new_tour)
                if new_dist < best_dist:
                    best_tour = new_tour.copy()
                    best_dist = new_dist
                    report_best_tour(best_tour)
                stagnation = 0
    return best_tour