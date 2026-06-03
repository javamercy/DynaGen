import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # Nearest neighbor initial tour
    unvisited = set(range(1, n))
    tour = [0]
    cur = 0
    while unvisited:
        next_city = min(unvisited, key=lambda x: distance_matrix[cur, x])
        tour.append(next_city)
        unvisited.remove(next_city)
        cur = next_city
    tour = np.array(tour, dtype=int)
    
    def compute_dist(t):
        s = 0.0
        for i in range(n):
            s += distance_matrix[t[i], t[(i+1)%n]]
        return s
    
    best_tour = tour.copy()
    best_dist = compute_dist(best_tour)
    report_best_tour(best_tour)  # initial report
    
    # First-improvement 2-opt local search
    def two_opt(t):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a, b = t[i], t[(i+1)%n]
                    c, d = t[j], t[(j+1)%n]
                    if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                        t[i+1:j+1] = t[i+1:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return t
    
    # Double-bridge perturbation
    def perturb(t):
        t = t.copy()
        i, j, k = sorted(np.random.choice(range(1, n-1), 3, replace=False))
        new_t = np.concatenate([t[:i], t[j:k], t[i:j], t[k:]])
        return new_t
    
    # Initial improvement
    current_tour = two_opt(tour)
    current_dist = compute_dist(current_tour)
    if current_dist < best_dist:
        best_dist = current_dist
        best_tour = current_tour.copy()
        report_best_tour(best_tour)
    
    # ILS iterations
    max_iter = 100
    for _ in range(max_iter):
        new_tour = perturb(current_tour)
        new_tour = two_opt(new_tour)
        new_dist = compute_dist(new_tour)
        if new_dist < best_dist:
            best_dist = new_dist
            best_tour = new_tour.copy()
            report_best_tour(best_tour)
        current_tour = new_tour
        current_dist = new_dist
    
    return best_tour