import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    # initial tour from first three cities
    tour = [0, 1, 2]
    remaining = set(range(3, n))
    # insertion cost function
    def delta(city, pos):
        before = tour[pos-1]
        after = tour[pos] if pos < len(tour) else tour[0]
        return distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
    # insert all remaining cities using regret heuristic
    while remaining:
        best_city = -1
        best_regret = -1
        best_pos = -1
        best_cost = float('inf')
        for city in remaining:
            costs = []
            for pos in range(len(tour)):
                costs.append((delta(city, pos), pos))
            costs.sort(key=lambda x: x[0])
            best = costs[0][0]
            second_best = costs[1][0] if len(costs) > 1 else best
            regret = second_best - best
            if regret > best_regret or (regret == best_regret and city < best_city):
                best_regret = regret
                best_city = city
                best_pos = costs[0][1]
                best_cost = best
        tour.insert(best_pos, best_city)
        remaining.remove(best_city)
    tour_arr = np.array(tour)
    report_best_tour(tour_arr)
    # 2-opt improvement with iteration budget
    max_passes = n  # adaptive budget: at most n full passes
    for _ in range(max_passes):
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                if j - i < n:
                    # consider reversing segment i+1..j
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    old = distance_matrix[a,b] + distance_matrix[c,d]
                    new = distance_matrix[a,c] + distance_matrix[b,d]
                    if new < old - 1e-10:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        tour_arr = np.array(tour)
                        report_best_tour(tour_arr)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return np.array(tour)