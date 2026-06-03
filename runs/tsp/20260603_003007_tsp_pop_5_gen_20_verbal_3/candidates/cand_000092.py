import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    best_dist = np.inf
    best_tour = None
    
    def tour_distance(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    
    def two_opt(tour):
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    if j - i == 1:
                        continue
                    a, b = tour[i], tour[i+1]
                    c, d = tour[j], tour[(j+1)%n]
                    if distance_matrix[a,c] + distance_matrix[b,d] < distance_matrix[a,b] + distance_matrix[c,d]:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        return tour
    
    # Multiple restarts
    for _ in range(10):
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
        tour = np.array(tour)
        tour = two_opt(tour)
        dist = tour_distance(tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
    
    # Iterated local search from best
    for _ in range(5):
        i, j = np.random.choice(n, 2, replace=False)
        perturbed = best_tour.copy()
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        perturbed = two_opt(perturbed)
        dist = tour_distance(perturbed)
        if dist < best_dist:
            best_dist = dist
            best_tour = perturbed.copy()
            report_best_tour(best_tour)
    
    return best_tour