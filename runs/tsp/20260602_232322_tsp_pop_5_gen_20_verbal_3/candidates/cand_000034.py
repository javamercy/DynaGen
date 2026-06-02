import numpy as np
import random

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    best_tour = None
    best_dist = float('inf')

    for restart in range(min(5, n)):
        # Regret insertion construction
        start = random.sample(range(n), 3)
        tour = list(start)
        remaining = set(range(n)) - set(start)

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

        tour_arr = np.array(tour)
        dist = 0
        for i in range(n):
            dist += distance_matrix[tour[i], tour[(i+1)%n]]
        if dist < best_dist - 1e-10:
            best_dist = dist
            best_tour = tour_arr.copy()
            report_best_tour(best_tour)

        # Iterated local search with 2-opt and double-bridge
        for iteration in range(10):
            # 2-opt local search
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    for j in range(i+2, n):
                        if j - i == 1:
                            continue
                        delta = (distance_matrix[tour[i], tour[j]] +
                                 distance_matrix[tour[i+1], tour[(j+1)%n]] -
                                 distance_matrix[tour[i], tour[i+1]] -
                                 distance_matrix[tour[j], tour[(j+1)%n]])
                        if delta < -1e-10:
                            tour[i+1:j+1] = reversed(tour[i+1:j+1])
                            improved = True
                            dist += delta
                            if dist < best_dist - 1e-10:
                                best_dist = dist
                                best_tour = np.array(tour)
                                report_best_tour(best_tour)

            # Double-bridge perturbation
            if n >= 8:
                # choose 4 cut points such that segments have at least 2 nodes
                a = random.randint(1, n//4)
                b = random.randint(a+2, n//2)
                c = random.randint(b+2, 3*n//4)
                d = n  # end
                # segments: [0:a], [a:b], [b:c], [c:d]
                new_tour = (tour[0:a] + tour[c:d] + tour[b:c] + tour[a:b])
            else:
                # swap two random non-adjacent cities
                i, j = random.sample(range(n), 2)
                if abs(i-j) > 1 and (i != 0 or j != n-1):
                    tour[i], tour[j] = tour[j], tour[i]
                new_tour = tour
            tour = new_tour
            dist = 0
            for i in range(n):
                dist += distance_matrix[tour[i], tour[(i+1)%n]]
            # Note: after perturbation, the tour may be worse; we don't accept it directly, but 2-opt will improve

    return best_tour