import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])
    if n == 2:
        return np.array([0, 1])
    
    best_tour = None
    best_dist = float('inf')
    
    def tour_distance(tour):
        d = 0.0
        for i in range(n):
            d += distance_matrix[tour[i]][tour[(i+1)%n]]
        return d
    
    num_restarts = 5
    for _ in range(num_restarts):
        # random initial pair
        i, j = np.random.choice(n, 2, replace=False)
        tour = [int(i), int(j)]
        unvisited = set(range(n)) - {i, j}
        
        # regret insertion
        while unvisited:
            best_regret = -1
            best_city = None
            best_pos = None
            L = len(tour)
            for city in unvisited:
                costs = []
                for pos in range(L):
                    a = tour[pos]
                    b = tour[(pos+1)%L]
                    cost = distance_matrix[a][city] + distance_matrix[city][b] - distance_matrix[a][b]
                    costs.append(cost)
                sorted_costs = sorted(costs)
                best = sorted_costs[0]
                second_best = sorted_costs[1] if len(sorted_costs) > 1 else float('inf')
                regret = second_best - best
                if regret > best_regret:
                    best_regret = regret
                    best_city = city
                    best_pos = costs.index(best)
            tour.insert(best_pos+1, best_city)
            unvisited.remove(best_city)
        
        tour = np.array(tour, dtype=int)
        dist = tour_distance(tour)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour.copy()
            report_best_tour(best_tour)
        
        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            L = n
            for i in range(L):
                for j in range(i+2, L):
                    a = tour[i]
                    b = tour[(i+1)%L]
                    c = tour[j%L]
                    d = tour[(j+1)%L]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new = distance_matrix[a][c] + distance_matrix[b][d]
                    if new < old:
                        # reverse segment between i+1 and j
                        if i+1 < j+1:
                            tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        else:
                            # wrap around case; simpler: reconstruct tour after reversal
                            # but for L large enough, j > i+2, no wrap.
                            pass
                        new_dist = tour_distance(tour)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            best_tour = tour.copy()
                            report_best_tour(best_tour)
                        improved = True
    return best_tour