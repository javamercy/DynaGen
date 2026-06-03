import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 3:
        return np.arange(n)

    def tour_dist(tour):
        total = 0.0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1) % n]]
        return total

    def two_opt(tour):
        best = tour.copy()
        best_dist = tour_dist(best)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+2, n):
                    a, b = best[i], best[(i+1)%n]
                    c, d = best[j], best[(j+1)%n]
                    old = distance_matrix[a,b] + distance_matrix[c,d]
                    new = distance_matrix[a,c] + distance_matrix[b,d]
                    if new < old:
                        new_tour = np.concatenate([best[:i+1], best[i+1:j+1][::-1], best[j+1:]])
                        new_dist = tour_dist(new_tour)
                        if new_dist < best_dist:
                            best = new_tour
                            best_dist = new_dist
                            improved = True
        return best, best_dist

    def regret_construction():
        tour = [0]
        first = np.argmin(distance_matrix[0,1:]) + 1
        tour.append(first)
        unvisited = set(range(n)) - set(tour)
        while unvisited:
            best_insert = {}
            second_best = {}
            for city in unvisited:
                best = float('inf')
                sec = float('inf')
                best_idx = None
                for pos in range(len(tour)):
                    i = tour[pos]
                    j = tour[(pos+1) % len(tour)]
                    cost = distance_matrix[i,city] + distance_matrix[city,j] - distance_matrix[i,j]
                    if cost < best:
                        sec = best
                        best = cost
                        best_idx = pos
                    elif cost < sec:
                        sec = cost
                best_insert[city] = (best_idx, best)
                second_best[city] = sec if sec != float('inf') else best
            regret = {c: second_best[c] - best_insert[c][1] for c in unvisited}
            chosen = max(unvisited, key=lambda c: (regret[c], -best_insert[c][1]))
            idx, _ = best_insert[chosen]
            tour.insert(idx+1, chosen)
            unvisited.remove(chosen)
        return np.array(tour)

    def random_reversal(tour):
        i = np.random.randint(0, n-2)
        j = np.random.randint(i+2, n)
        new_tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
        return new_tour

    # Construction
    tour = regret_construction()
    best_tour, best_dist = two_opt(tour)
    report_best_tour(best_tour)

    # ILS
    max_iter = max(5, n // 10)
    current_tour = best_tour.copy()
    current_dist = best_dist
    for _ in range(max_iter):
        perturbed = random_reversal(current_tour)
        new_tour, new_dist = two_opt(perturbed)
        if new_dist < best_dist:
            best_tour = new_tour.copy()
            best_dist = new_dist
            report_best_tour(best_tour)
        current_tour = new_tour
        current_dist = new_dist

    return best_tour