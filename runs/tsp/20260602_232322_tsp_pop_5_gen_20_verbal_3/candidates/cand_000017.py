import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour
    import random
    best_tour = None
    best_dist = float('inf')
    for restart in range(min(10, n)):
        # start with random triangle
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)
        # regret insertion
        def delta(city, pos):
            before = tour[pos-1]
            after = tour[pos] if pos < len(tour) else tour[0]
            return distance_matrix[before, city] + distance_matrix[city, after] - distance_matrix[before, after]
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
                first = costs[0][0]
                second = costs[1][0] if len(costs) > 1 else first
                regret = second - first
                if regret > best_regret or (regret == best_regret and city < best_city):
                    best_regret = regret
                    best_city = city
                    best_pos = costs[0][1]
                    best_cost = first
            tour.insert(best_pos, best_city)
            remaining.remove(best_city)
        # initial distance
        tour_arr = np.array(tour)
        dist = 0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)
        # 2-opt
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+2, n):
                    if j - i < n:
                        new_tour = tour.copy()
                        new_tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        a = tour[i]
                        b = tour[(i+1)%n]
                        c = tour[j]
                        d = tour[(j+1)%n]
                        old = distance_matrix[a,b] + distance_matrix[c,d]
                        new = distance_matrix[a,c] + distance_matrix[b,d]
                        if new < old - 1e-10:
                            tour = new_tour
                            tour_arr = np.array(tour)
                            dist = 0
                            for i2 in range(n):
                                dist += distance_matrix[tour[i2], tour[(i2+1)%n]]
                            if dist < best_dist - 1e-10:
                                best_dist = dist
                                best_tour = tour_arr.copy()
                                report_best_tour(best_tour)
                            improved = True
                            break
                if improved:
                    break
    return best_tour