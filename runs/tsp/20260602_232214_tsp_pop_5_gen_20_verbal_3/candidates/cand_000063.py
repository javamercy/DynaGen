import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    rng = np.random.default_rng()
    
    def tour_length(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    
    def two_opt(tour):
        improved = True
        best = tour.copy()
        best_dist = tour_length(best)
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = best[i]; b = best[i+1]; c = best[j]; d = best[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        new = best.copy()
                        new[i+1:j+1] = best[j:i:-1]
                        new_dist = best_dist + delta
                        if new_dist < best_dist:
                            best = new
                            best_dist = new_dist
                            improved = True
                            report_best_tour(best)
        return best, best_dist
    
    # Initial random tour
    tour = np.arange(n)
    rng.shuffle(tour)
    report_best_tour(tour)
    tour, dist = two_opt(tour)
    best_tour = tour.copy()
    best_dist = dist
    
    # ILS with double-bridge perturbation
    current_tour = best_tour.copy()
    max_iters = max(50, int(np.ceil(n/2)))
    for _ in range(max_iters):
        cuts = sorted(rng.choice(np.arange(1, n-1), size=4, replace=False))
        a, b, c, d = cuts
        perturbed = np.concatenate([current_tour[:a+1], current_tour[c+1:d+1], current_tour[b+1:c+1], current_tour[a+1:b+1], current_tour[d+1:]])
        if len(perturbed) != n:
            continue
        new_tour, new_dist = two_opt(perturbed)
        if new_dist < best_dist:
            best_tour = new_tour.copy()
            best_dist = new_dist
            report_best_tour(best_tour)
        current_tour = new_tour
    
    return best_tour