import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n == 1:
        return np.array([0])

    def tour_dist(tour):
        d = 0.0
        for i in range(n):
            d += distance_matrix[tour[i]][tour[(i+1)%n]]
        return d

    def two_opt(tour):
        improved = True
        best = tour[:]
        best_dist = tour_dist(best)
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = best[i]
                    b = best[(i+1)%n]
                    c = best[j%n]
                    d = best[(j+1)%n]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new = distance_matrix[a][c] + distance_matrix[b][d]
                    if new < old:
                        best[i+1:j+1] = reversed(best[i+1:j+1])
                        best_dist = best_dist - old + new
                        improved = True
        return best, best_dist

    # Regret insertion initialization
    max_dist = -1
    best_pair = (0,1)
    for i in range(n):
        for j in range(i+1, n):
            d = distance_matrix[i][j]
            if d > max_dist:
                max_dist = d
                best_pair = (i,j)
    tour = list(best_pair)
    unvisited = set(range(n)) - set(tour)
    while unvisited:
        best_regret = -1
        best_city = None
        best_pos = None
        for k in unvisited:
            costs = []
            L = len(tour)
            for pos in range(L):
                i = tour[pos]
                j = tour[(pos+1)%L]
                cost = distance_matrix[i][k] + distance_matrix[k][j] - distance_matrix[i][j]
                costs.append(cost)
            sorted_costs = sorted(costs)
            best_cost = sorted_costs[0]
            second_best = sorted_costs[1] if len(sorted_costs)>1 else float('inf')
            regret = second_best - best_cost
            if regret > best_regret:
                best_regret = regret
                best_city = k
                best_pos = costs.index(best_cost)
        tour.insert(best_pos+1, best_city)
        unvisited.remove(best_city)

    best_tour = tour[:]
    best_dist = tour_dist(best_tour)
    report_best_tour(np.array(best_tour))

    # Initial 2-opt
    best_tour, best_dist = two_opt(best_tour)
    report_best_tour(np.array(best_tour))

    # ILS with double-bridge and SA acceptance
    T = best_dist / 100.0
    current_tour = best_tour[:]
    current_dist = best_dist
    for _ in range(50):
        if n < 8:
            break
        indices = sorted(np.random.choice(range(1, n-1), 4, replace=False))
        a, b, c, d = indices
        perturbed = current_tour[:a] + current_tour[c:d] + current_tour[b:c] + current_tour[a:b] + current_tour[d:]
        if len(set(perturbed)) != n:
            continue
        new_tour, new_dist = two_opt(perturbed)
        delta = new_dist - current_dist
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_tour = new_tour
            current_dist = new_dist
            if new_dist < best_dist:
                best_dist = new_dist
                best_tour = new_tour
                report_best_tour(np.array(best_tour))
        T *= 0.99

    return np.array(best_tour)