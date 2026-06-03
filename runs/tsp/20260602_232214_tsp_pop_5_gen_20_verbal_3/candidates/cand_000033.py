import numpy as np

def solve_tsp(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 3:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    def tour_dist(tour):
        total = 0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i+1)%n]]
        return total

    def local_2opt(tour, best_dist):
        improved = True
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = tour[i]
                    b = tour[(i+1)%n]
                    c = tour[j]
                    d = tour[(j+1)%n]
                    old = distance_matrix[a,b] + distance_matrix[c,d]
                    new = distance_matrix[a,c] + distance_matrix[b,d]
                    if new < old:
                        new_tour = np.concatenate([tour[:i+1], tour[i+1:j+1][::-1], tour[j+1:]])
                        new_dist = tour_dist(new_tour)
                        if new_dist < best_dist:
                            best_dist = new_dist
                            tour = new_tour
                            improved = True
                            report_best_tour(tour)
                            break
                if improved:
                    break
        return tour, best_dist

    # Regret-2 construction with random tie-breaking
    tour = [0]
    first = np.argmin(distance_matrix[0][1:]) + 1
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
                j = tour[(pos+1)%len(tour)]
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
        max_regret = max(regret.values())
        candidates = [c for c in unvisited if regret[c] == max_regret]
        chosen = np.random.choice(candidates)
        idx, _ = best_insert[chosen]
        tour.insert(idx+1, chosen)
        unvisited.remove(chosen)

    best_tour = np.array(tour)
    best_dist = tour_dist(best_tour)
    report_best_tour(best_tour)

    # Initial 2-opt
    best_tour, best_dist = local_2opt(best_tour, best_dist)

    # Iterated local search with double-bridge perturbation
    for _ in range(30):
        # double-bridge perturbation
        indices = sorted(np.random.choice(range(1, n-1), size=4, replace=False))
        a, b, c, d = indices
        new_tour = np.concatenate([best_tour[:a], best_tour[c:d], best_tour[b:c], best_tour[a:b], best_tour[d:]])
        new_tour = np.array(new_tour, dtype=int)
        new_tour, new_dist = local_2opt(new_tour, best_dist)
        if new_dist < best_dist:
            best_tour = new_tour
            best_dist = new_dist
            report_best_tour(best_tour)

    return best_tour