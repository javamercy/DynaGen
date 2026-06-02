import numpy as np

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    
    # Regret insertion construction
    tour = [0, 1, 2]
    remaining = set(range(3, n))
    def delta(city, pos):
        before = tour[pos-1]
        after = tour[pos % len(tour)]
        return distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
    
    while remaining:
        best_city = -1
        best_regret = -1
        best_pos = -1
        for city in remaining:
            costs = [(delta(city, pos), pos) for pos in range(len(tour))]
            costs.sort()
            best = costs[0][0]
            second_best = costs[1][0] if len(costs) > 1 else best
            regret = second_best - best
            if regret > best_regret:
                best_regret = regret
                best_city = city
                best_pos = costs[0][1]
        tour.insert(best_pos, best_city)
        remaining.remove(best_city)
    
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    
    # 2-opt improvement until local optimum
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b = tour[i], tour[(i+1)%n]
                c, d = tour[j], tour[(j+1)%n]
                if distance_matrix[a, c] + distance_matrix[b, d] < distance_matrix[a, b] + distance_matrix[c, d]:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    tour_arr = np.array(tour)
                    report_best_tour(tour_arr)
                    improved = True
                    break
            if improved:
                break
    
    return np.array(tour)