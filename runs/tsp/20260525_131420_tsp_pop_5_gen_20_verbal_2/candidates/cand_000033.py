import numpy as np

def solve_tsp(distance_matrix: np.ndarray, seed: int, budget: int) -> np.ndarray:
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    
    def nearest_neighbor(start):
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        last = start
        while unvisited:
            best_city = min(unvisited, key=lambda city: distance_matrix[last, city])
            tour.append(best_city)
            unvisited.remove(best_city)
            last = best_city
        return np.array(tour, dtype=np.int32)
    
    best_tour = None
    best_dist = float('inf')
    
    start = np.random.randint(n)
    tour = nearest_neighbor(start)
    total = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    best_tour = tour.copy()
    best_dist = total
    report_best_tour(best_tour)
    
    iteration = 0
    while iteration < budget:
        improved = False
        for i in range(n-2):
            for j in range(i+2, n):
                if iteration >= budget:
                    break
                a, b, c, d = tour[i], tour[i+1], tour[j], tour[(j+1)%n]
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-12:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    total += delta
                    if total < best_dist - 1e-12:
                        best_dist = total
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    improved = True
                    break
            if improved or iteration >= budget:
                break
        if not improved:
            if iteration >= budget:
                break
            # perturbation: random segment inversion
            i = np.random.randint(0, n-2)
            j = np.random.randint(i+2, n)
            tour[i+1:j+1] = tour[i+1:j+1][::-1]
            total = sum(distance_matrix[tour[k], tour[(k+1)%n]] for k in range(n))
            improved = True
        iteration += 1
    
    return best_tour