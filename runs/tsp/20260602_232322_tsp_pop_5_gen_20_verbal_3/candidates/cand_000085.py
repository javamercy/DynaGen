import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    best_tour = None
    best_dist = np.inf
    for _ in range(20):
        start = random.randrange(n)
        tour = [start]
        remaining = set(range(n)) - {start}
        while remaining:
            best_city = -1
            best_regret = -1.0
            best_pos = -1
            for city in remaining:
                costs = []
                for pos in range(len(tour)):
                    before = tour[pos - 1]
                    after = tour[pos]
                    delta = distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
                    costs.append((delta, pos))
                costs.sort(key=lambda x: x[0])
                first = costs[0][0]
                second = costs[1][0] if len(costs) > 1 else first
                regret = second - first
                if regret > best_regret or (regret == best_regret and city < best_city):
                    best_regret = regret
                    best_city = city
                    best_pos = costs[0][1]
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)
        tour_arr = np.array(tour, dtype=np.int64)
        dist = 0.0
        for i in range(n):
            dist += distance_matrix[tour_arr[i], tour_arr[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)
        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a, b, c, d = tour[i], tour[(i+1)%n], tour[j], tour[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        dist += delta
                        if dist < best_dist - 1e-10:
                            best_dist = dist
                            best_tour = np.array(tour)
                            report_best_tour(best_tour)
                        improved = True
                        break
                if improved:
                    break
        # Note: tour is modified in-place, but we use it for next restart? Actually we want fresh restart each time.
        # So we reassign tour at start of loop; after 2-opt, we don't reuse tour for next iteration.
    return best_tour