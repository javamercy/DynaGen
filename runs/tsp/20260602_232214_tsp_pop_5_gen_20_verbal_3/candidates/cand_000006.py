import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.array(range(n))
        report_best_tour(tour)
        return tour
    
    def tour_distance(tour):
        total = 0.0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1)%n]]
        return total
    
    # random initial tour
    best_tour = np.random.permutation(n)
    best_dist = tour_distance(best_tour)
    report_best_tour(best_tour)
    
    current_tour = best_tour.copy()
    current_dist = best_dist
    
    # simulated annealing parameters
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.995
    
    while T > T_min:
        # generate neighbor by swapping two random cities
        i, j = np.random.randint(n), np.random.randint(n)
        while i == j:
            j = np.random.randint(n)
        
        new_tour = current_tour.copy()
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_dist = tour_distance(new_tour)
        delta = new_dist - current_dist
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_tour = new_tour
            current_dist = new_dist
            if current_dist < best_dist:
                best_tour = current_tour.copy()
                best_dist = current_dist
                report_best_tour(best_tour)
        
        T *= cooling_rate
    
    return best_tour