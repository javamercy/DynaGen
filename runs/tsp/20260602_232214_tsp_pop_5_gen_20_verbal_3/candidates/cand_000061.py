import numpy as np

def solve_tsp(distance_matrix: np.ndarray) -> np.ndarray:
    n = len(distance_matrix)
    if n <= 2:
        tour = np.arange(n)
        report_best_tour(tour)
        return tour

    rng = np.random.default_rng()

    def tour_length(tour):
        return sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))

    def two_opt(tour):
        improved = True
        best = tour.copy()
        best_dist = tour_length(best)
        while improved:
            improved = False
            for i in range(n-2):
                for j in range(i+2, n):
                    a = best[i]
                    b = best[i+1]
                    c = best[j]
                    d = best[(j+1)%n]
                    delta = distance_matrix[a,c] + distance_matrix[b,d] - distance_matrix[a,b] - distance_matrix[c,d]
                    if delta < -1e-10:
                        new = best.copy()
                        new[i+1:j+1] = best[j:i:-1]
                        new_dist = best_dist + delta
                        if new_dist < best_dist - 1e-10:
                            best = new
                            best_dist = new_dist
                            improved = True
                            report_best_tour(best)
        return best, best_dist

    def regret_insertion():
        i, j = np.unravel_index(np.argmax(distance_matrix), (n, n))
        tour = [i, j]
        unvisited = set(range(n)) - {i, j}
        while unvisited:
            best_cost = {}
            second_cost = {}
            best_pos = {}
            for city in unvisited:
                best = float('inf')
                second = float('inf')
                pos = 0
                for k in range(len(tour)):
                    a = tour[k]
                    b = tour[(k+1)%len(tour)]
                    cost = distance_matrix[a, city] + distance_matrix[city, b] - distance_matrix[a, b]
                    if cost < best:
                        second = best
                        best = cost
                        pos = k+1
                    elif cost < second:
                        second = cost
                best_cost[city] = best
                second_cost[city] = second if second != float('inf') else best
                best_pos[city] = pos
            max_regret = max(second_cost[c] - best_cost[c] for c in unvisited)
            candidates = [c for c in unvisited if second_cost[c] - best_cost[c] == max_regret]
            if len(candidates) > 1:
                chosen = min(candidates, key=lambda c: best_cost[c])
            else:
                chosen = candidates[0]
            pos = best_pos[chosen]
            tour.insert(pos, chosen)
            unvisited.remove(chosen)
        return np.array(tour)

    best_tour = regret_insertion()
    report_best_tour(best_tour)
    best_tour, best_dist = two_opt(best_tour)

    # ILS with double-bridge
    max_iters = max(10, int(np.ceil(n/10)))
    for _ in range(max_iters):
        cuts = sorted(rng.choice(np.arange(1, n-1), size=4, replace=False))
        a, b, c, d = cuts[0], cuts[1], cuts[2], cuts[3]
        perturbed = np.concatenate([best_tour[:a+1], best_tour[c+1:d+1], best_tour[b+1:c+1], best_tour[a+1:b+1], best_tour[d+1:]])
        if len(perturbed) != n:
            continue
        new_tour, new_dist = two_opt(perturbed)
        if new_dist < best_dist:
            best_tour = new_tour
            best_dist = new_dist
            report_best_tour(best_tour)

    return best_tour