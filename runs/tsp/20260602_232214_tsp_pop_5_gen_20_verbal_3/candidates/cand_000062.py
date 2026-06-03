import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    # Regret insertion initial tour
    tour = [0, 1]
    unvisited = set(range(2, n))
    
    def insertion_cost(tour, city):
        best = float('inf')
        second_best = float('inf')
        best_idx = 0
        for idx in range(len(tour)):
            a = tour[idx]
            b = tour[(idx + 1) % len(tour)]
            cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
            if cost < best:
                second_best = best
                best = cost
                best_idx = idx + 1
            elif cost < second_best:
                second_best = cost
        return best, second_best, best_idx
    
    while unvisited:
        best_city = None
        max_regret = -float('inf')
        best_insert_idx = None
        for city in unvisited:
            best_cost, second_best, insert_idx = insertion_cost(tour, city)
            regret = second_best - best_cost
            if regret > max_regret:
                max_regret = regret
                best_city = city
                best_insert_idx = insert_idx
        tour.insert(best_insert_idx, best_city)
        unvisited.remove(best_city)
    
    tour = np.array(tour)
    best_tour = tour.copy()
    best_dist = np.sum(distance_matrix[tour, np.roll(tour, -1)])
    report_best_tour(best_tour)
    
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b, c, d = tour[i], tour[(i + 1) % n], tour[j], tour[(j + 1) % n]
                old_dist = distance_matrix[a, b] + distance_matrix[c, d]
                new_dist = distance_matrix[a, c] + distance_matrix[b, d]
                if new_dist < old_dist:
                    # reverse segment i+1 .. j
                    tour[i + 1:j + 1] = np.flip(tour[i + 1:j + 1])
                    improved = True
                    new_dist_total = np.sum(distance_matrix[tour, np.roll(tour, -1)])
                    if new_dist_total < best_dist:
                        best_dist = new_dist_total
                        best_tour = tour.copy()
                        report_best_tour(best_tour)
                    break
            if improved:
                break
    return best_tour